# ==========================================================
# pipeline.py
# ==========================================================

from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import json
import os


from utilities.loggers import LoggerObserver, FileLogger, ImageWriter, WandbLogger, WandbFigureLogger

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path


# ==========================================================
# LOCAL IMPORTS
# ==========================================================

from registry import Registry

from datasets import (
    DATASET_REGISTRY,
)

from encoders import (
    ENCODER_REGISTRY,
)

from losses import (
    LOSS_REGISTRY,
)

from metrics import (
    METRIC_REGISTRY,
)

from models import (
    MODEL_REGISTRY,
)

from odes import (
    ODE_REGISTRY,
)

from solvers import (
    SOLVER_REGISTRY,
)

from trainer import (
    TRAINER_REGISTRY,
)

from utilities import (
    seed_everything,
    plot_parameter_recovery,
    plot_training_history,
    plot_trajectory_examples,
    build_optimizer
)


# ==========================================================
# BASE PIPELINE
# ==========================================================


class BasePipeline:
    """
    Main pipeline for Physiome ODE parameter estimation.

    Architecture:

        Hydra config
            |
            v
        Registries
            |
            +--> Dataset / DataModule
            +--> Encoder
            +--> ODE
            +--> Solver
            +--> Model
            +--> Loss
            +--> Metrics
            +--> Optimizer
            +--> Trainer
            |
            v
        Training
            |
            v
        Evaluation
            |
            v
        Visualization

    All components are selected through configuration.
    """

    # ======================================================
    # INITIALIZATION
    # ======================================================

    def __init__(self, config: DictConfig) -> None:

        self.config = config

        # --------------------------------------------------
        # Initialize registries
        # --------------------------------------------------

        self.init_registry()

        # --------------------------------------------------
        # Global configuration
        # --------------------------------------------------

        self.init_globals()

        self.log("\n" + "=" * 70)
        self.log("CONFIGURATION")
        self.log("=" * 70)
        self.log(OmegaConf.to_yaml(self.config))

    def log(self, *values, level=None) -> None:
        """Send a message to the configured observer subscribers."""
        message = " ".join(str(value) for value in values)
        if level is None:
            level = LoggerObserver.INFO
        self.logger.text(message, level=level)

    def log_metrics(self, values: Dict[str, Any], prefix: str = "test") -> None:
        """Send scalar metrics to FileLogger/W&B through LoggerObserver."""
        events = []

        def flatten(obj, name):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    flatten(value, f"{name}/{key}" if name else str(key))
            elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
                events.append({"type": LoggerObserver.SCALAR,
                              "tag": f"{prefix}/{name}", "value": float(obj), "step": 0})
        flatten(values, "")
        if events:
            self.logger.log(events)

    def upload_figures_to_wandb(self, directory=None):
        """
        Upload every locally saved image in the figure directory.

        Calling this method multiple times is safe: already-uploaded
        unchanged files are skipped.
        """
        if not hasattr(self, "figure_logger"):
            self.log(
                "[WARNING] Figure logger has not been initialized."
            )
            return 0

        return self.figure_logger.upload_all(directory=directory)

    # ======================================================
    # GLOBAL CONFIGURATION
    # ======================================================

    def init_globals(self) -> None:
        """
        Initialize global settings such as random seed,
        device and output directories.
        """
        # One folder per run; all output paths are redirected there.
        self.logger = LoggerObserver.getLogger("main")
        self.figure_logger = WandbFigureLogger(
            figure_dir=self.figure_dir,
            logger=self.logger,
        )
        configured_root = self.config.output.get("run_dir", "runs")
        run_root = Path(to_absolute_path(str(configured_root)))
        run_name = str(self.config.get("run_name", "physiome_ode"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = run_root / f"{run_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir = self.run_dir / "figures"
        self.figure_dir.mkdir(parents=True, exist_ok=True)

        # Logger configuration is controlled by the YAML `logging` section.
        # Keep FileLogger and ImageWriter from the existing project utilities.
        logging_cfg = self.config.get("logging", {})
        file_cfg = logging_cfg.get("file", {})
        image_cfg = logging_cfg.get("image", {})
        wandb_cfg = logging_cfg.get("wandb", {})

        if bool(file_cfg.get("enabled", True)):
            self.logger.subscribe(
                FileLogger(
                    __name__,
                    str(self.run_dir),
                    debug=bool(file_cfg.get("debug", False)),
                )
            )

        if bool(image_cfg.get("enabled", True)):
            self.logger.subscribe(ImageWriter(str(self.run_dir)))

        config_dict = OmegaConf.to_container(self.config, resolve=True)
        (self.run_dir / "config.json").write_text(
            json.dumps(config_dict, indent=2, default=str), encoding="utf-8"
        )

        # Do not start a W&B run unless it is explicitly enabled and a project
        # name is configured. API keys should be supplied via WANDB_API_KEY.
        wandb_project = wandb_cfg.get(
            "project_name") or os.getenv("WANDB_PROJECT")
        if bool(wandb_cfg.get("enabled", False)) and wandb_project:
            wandb_run_name = wandb_cfg.get("run_name") or run_name
            wandb_logger = WandbLogger(
                unique_id=wandb_cfg.get("unique_id", None),
                username=wandb_cfg.get("entity") or os.getenv("WANDB_ENTITY"),
                project_name=str(wandb_project),
                run_name=str(wandb_run_name),
                group_name=wandb_cfg.get("group_name", None),
                save_dir=str(self.run_dir),
                config_dict=config_dict,
                tags=wandb_cfg.get("tags", None),
                reinit=bool(wandb_cfg.get("reinit", True)),
            )
            self.logger.subscribe(wandb_logger)

        self.log(f"Run directory: {self.run_dir}")

        # --------------------------------------------------
        # Seed
        # --------------------------------------------------

        seed = int(self.config.experiment.seed)

        seed_everything(seed)

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        requested_device = str(self.config.experiment.device)

        if requested_device == "cuda" and not torch.cuda.is_available():
            self.log(
                "[WARNING] CUDA requested but not available. "
                "Falling back to CPU."
            )

            self.device = torch.device("cpu")

        else:
            self.device = torch.device(requested_device)
            # self.device = torch.device("cuda:0")

        self.log(f"[INFO] Seed   : {seed}")
        self.log(f"[INFO] Device : {self.device}")

        # All artifacts live in this run's directory, avoiding overwrites.
        self.checkpoint_path = self.run_dir / \
            Path(str(self.config.output.checkpoint)).name
        self.results_path = self.run_dir / \
            Path(str(self.config.output.results)).name
        self.history_path = self.run_dir / \
            Path(str(self.config.output.history)).name

        self.checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.results_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.history_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.figure_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ======================================================
    # REGISTRIES
    # ======================================================

    def init_registry(self) -> None:
        """
        Attach all component registries.

        Registry keys are the Python class names because
        registration is done without explicit names.

        Example:

            ENCODER_REGISTRY.register(GRUEncoder)

        gives:

            "GRUEncoder" -> GRUEncoder
        """

        # --------------------------------------------------
        # Data
        # --------------------------------------------------

        self.dataset_registry = DATASET_REGISTRY

        # --------------------------------------------------
        # Other components
        # --------------------------------------------------

        self.encoder_registry = ENCODER_REGISTRY
        self.loss_registry = LOSS_REGISTRY
        self.metric_registry = METRIC_REGISTRY
        self.model_registry = MODEL_REGISTRY
        self.ode_registry = ODE_REGISTRY
        self.solver_registry = SOLVER_REGISTRY
        self.trainer_registry = TRAINER_REGISTRY

    # ======================================================
    # DATA
    # ======================================================

    def init_data(self) -> None:
        """
        Initialize the configured DataModule.
        """

        self.log("[1/10] DATA")

        # --------------------------------------------------
        # Read name from config
        # --------------------------------------------------

        datamodule_name = str(
            self.config.data.name
        )

        self.log(
            f"       DataModule: {datamodule_name}"
        )

        # --------------------------------------------------
        # Resolve registry entry
        # --------------------------------------------------

        datamodule_cls = (
            self.dataset_registry.get(
                datamodule_name
            )
        )

        # --------------------------------------------------
        # Data path
        # --------------------------------------------------

        data_path = to_absolute_path(
            str(self.config.data.path)
        )

        # --------------------------------------------------
        # Instantiate DataModule
        # --------------------------------------------------

        self.datamodule = datamodule_cls(
            data_path=data_path,
            batch_size=int(
                self.config.data.batch_size
            ),
            num_workers=int(
                self.config.data.num_workers
            ),
        )

        # --------------------------------------------------
        # Setup
        # --------------------------------------------------

        self.datamodule.setup()

        # --------------------------------------------------
        # DataLoaders
        # --------------------------------------------------

        self.train_loader = (
            self.datamodule.train_loader()
        )

        self.valid_loader = (
            self.datamodule.valid_loader()
        )

        self.test_loader = (
            self.datamodule.test_loader()
        )

        self.log(
            f"       Data path : {data_path}"
        )

        self.log(
            f"       Batch size: "
            f"{self.config.data.batch_size}"
        )

        self.log("       DATA OK")

    # ======================================================
    # EDA
    # ======================================================

    def run_eda(self) -> None:
        """
        Run dataset exploratory data analysis.

        Visualization/EDA logic stays outside the trainer.
        """

        self.log("[2/10] EDA")

        try:
            from datasets.eda import DatasetEDA

        except ImportError as exc:
            self.log(
                "[WARNING] DatasetEDA could not be imported:"
            )
            self.log(f"          {exc}")
            self.log("          Skipping EDA.")
            return

        eda_output_dir = (
            self.figure_dir / "eda"
        )

        eda = DatasetEDA(
            output_dir=str(eda_output_dir),
            logger=self.logger,
        )

        # --------------------------------------------------
        # Analyze datasets
        # --------------------------------------------------

        if hasattr(self.datamodule, "train_dataset"):
            train_dataset = (
                self.datamodule.train_dataset
            )
        else:
            train_dataset = None

        if hasattr(self.datamodule, "valid_dataset"):
            valid_dataset = (
                self.datamodule.valid_dataset
            )
        elif hasattr(self.datamodule, "val_dataset"):
            valid_dataset = (
                self.datamodule.val_dataset
            )
        else:
            valid_dataset = None

        if hasattr(self.datamodule, "test_dataset"):
            test_dataset = (
                self.datamodule.test_dataset
            )
        else:
            test_dataset = None

        # --------------------------------------------------
        # Train
        # --------------------------------------------------

        if train_dataset is not None:
            eda.analyze(
                train_dataset,
                dataset_name="train",
            )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        if valid_dataset is not None:
            eda.analyze(
                valid_dataset,
                dataset_name="valid",
            )

        # --------------------------------------------------
        # Test
        # --------------------------------------------------

        if test_dataset is not None:
            eda.analyze(
                test_dataset,
                dataset_name="test",
            )

        self.upload_figures_to_wandb(
            directory=eda_output_dir
        )

        self.log("       EDA OK")

    # ======================================================
    # ENCODER
    # ======================================================

    def init_encoder(self) -> None:
        """
        Initialize the selected parameter encoder.
        """

        self.log("[3/10] ENCODER")

        encoder_name = str(
            self.config.encoder.name
        )

        self.log(
            f"       Encoder: {encoder_name}"
        )

        encoder_cls = (
            self.encoder_registry.get(
                encoder_name
            )
        )

        # --------------------------------------------------
        # Input dimension
        #
        # Features:
        #
        #   time
        #   state values
        #   observation mask
        #
        # => 1 + D + D
        # --------------------------------------------------

        state_dim = int(
            self.config.model.state_dim
        )

        input_dim = (
            1
            + state_dim
            + state_dim
        )

        self.log(
            f"       Input dim: {input_dim}"
        )

        # --------------------------------------------------
        # Encoder parameters
        # --------------------------------------------------

        encoder_params = OmegaConf.to_container(
            self.config.encoder.params,
            resolve=True,
        )

        if encoder_params is None:
            encoder_params = {}

        # --------------------------------------------------
        # Instantiate
        # --------------------------------------------------

        self.encoder = encoder_cls(
            input_dim=input_dim,
            state_dim=state_dim,
            parameter_dim=int(
                self.config.model.parameter_dim
            ),
            **encoder_params,
        )

        self.log("       ENCODER OK")

    # ======================================================
    # ODE
    # ======================================================

    def init_ode(self) -> None:
        """
        Initialize the configured ODE model.
        """

        self.log("[4/10] ODE")

        ode_name = str(
            self.config.ode.name
        )

        self.log(
            f"       ODE: {ode_name}"
        )

        ode_cls = (
            self.ode_registry.get(
                ode_name
            )
        )

        # --------------------------------------------------
        # Convert configuration to Python objects
        # --------------------------------------------------

        parameters = OmegaConf.to_container(
            self.config.ode.parameters,
            resolve=True,
        )

        initial_state = OmegaConf.to_container(
            self.config.ode.initial_state,
            resolve=True,
        )

        parameter_names = list(
            self.config.ode.parameter_names
        )

        state_names = list(
            self.config.ode.state_names
        )

        # --------------------------------------------------
        # Instantiate ODE
        # --------------------------------------------------

        self.ode = ode_cls(
            parameters=parameters,
            initial_state=initial_state,
            parameter_names=parameter_names,
            state_names=state_names,
        )

        self.log("       ODE OK")

    # ======================================================
    # SOLVER
    # ======================================================

    def init_solver(self) -> None:
        """
        Initialize the configured ODE solver.
        """

        self.log("[5/10] SOLVER")

        solver_name = str(
            self.config.solver.name
        )

        self.log(
            f"       Solver: {solver_name}"
        )

        solver_cls = (
            self.solver_registry.get(
                solver_name
            )
        )

        solver_params = OmegaConf.to_container(
            self.config.solver.params,
            resolve=True,
        )

        if solver_params is None:
            solver_params = {}

        # --------------------------------------------------
        # Instantiate
        # --------------------------------------------------

        self.solver = solver_cls(
            **solver_params
        )

        self.log("       SOLVER OK")

    # ======================================================
    # MODEL
    # ======================================================

    def init_model(self) -> None:
        """
        Build the high-level ParameterEstimator by
        composing encoder + ODE + solver.
        """

        self.log("[6/10] MODEL")

        model_name = str(
            self.config.model.name
        )

        self.log(
            f"       Model: {model_name}"
        )

        model_cls = (
            self.model_registry.get(
                model_name
            )
        )

        # --------------------------------------------------
        # Instantiate
        # --------------------------------------------------

        self.model = model_cls(
            encoder=self.encoder,
            ode=self.ode,
            solver=self.solver
        )

        self.model.to(self.device)

        self.log(
            f"       Parameters: "
            f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}"
        )

        self.log("       MODEL OK")

    # ======================================================
    # LOSS
    # ======================================================

    def init_loss(self) -> None:
        """
        Initialize the configured loss function.
        """

        self.log("[7/10] LOSS")

        loss_name = str(
            self.config.loss.name
        )

        self.log(
            f"       Loss: {loss_name}"
        )

        loss_cls = (
            self.loss_registry.get(
                loss_name
            )
        )

        # --------------------------------------------------
        # Loss parameters
        # --------------------------------------------------

        parameter_loss_weight = float(
            self.config.training.parameter_loss_weight
        )

        trajectory_loss_weight = float(
            self.config.training.trajectory_loss_weight
        )

        # --------------------------------------------------
        # Instantiate
        # --------------------------------------------------

        self.loss_fn = loss_cls(
            parameter_loss_weight=(
                parameter_loss_weight
            ),
            trajectory_loss_weight=(
                trajectory_loss_weight
            ),
        )

        self.log("       LOSS OK")

    # ======================================================
    # METRICS
    # ======================================================

    def init_metrics(self) -> None:
        """
        Initialize parameter, trajectory and JGD metrics.
        """

        self.log("[8/10] METRICS")

        # --------------------------------------------------
        # Parameter metrics
        # --------------------------------------------------

        self.parameter_metrics = None

        if self.config.metrics.parameter.enabled:

            parameter_metric_methods = list(
                self.config.metrics.parameter.methods
            )

            parameter_metric_params = OmegaConf.to_container(
                self.config.metrics.parameter.params,
                resolve=True,
            )

            parameter_metric_cls = (
                self.metric_registry.get(
                    "ParameterMetrics"
                )
            )

            self.parameter_metrics = (
                parameter_metric_cls(
                    methods=parameter_metric_methods,
                    **parameter_metric_params,
                )
            )

        # --------------------------------------------------
        # Trajectory metrics
        # --------------------------------------------------

        self.trajectory_metrics = None

        if self.config.metrics.trajectory.enabled:

            trajectory_metric_methods = list(
                self.config.metrics.trajectory.methods
            )

            trajectory_metric_params = OmegaConf.to_container(
                self.config.metrics.trajectory.params,
                resolve=True,
            )

            trajectory_metric_cls = (
                self.metric_registry.get(
                    "TrajectoryMetrics"
                )
            )

            self.trajectory_metrics = (
                trajectory_metric_cls(
                    methods=trajectory_metric_methods,
                    **trajectory_metric_params,
                )
            )

        # --------------------------------------------------
        # JGD metrics
        # --------------------------------------------------

        self.jgd_metrics = None

        if self.config.metrics.jgd.enabled:

            jgd_metric_methods = list(
                self.config.metrics.jgd.methods
            )

            jgd_metric_params = OmegaConf.to_container(
                self.config.metrics.jgd.params,
                resolve=True,
            )

            jgd_metric_cls = (
                self.metric_registry.get(
                    "JGDMetrics"
                )
            )

            self.jgd_metrics = (
                jgd_metric_cls(
                    methods=jgd_metric_methods,
                    **jgd_metric_params,
                )
            )

        self.log("       METRICS OK")

    # ======================================================
    # OPTIMIZER
    # ======================================================

    def init_optimizer(self) -> None:
        """
        Initialize the optimizer directly from configuration.

        There is intentionally no optimizer registry.
        """

        self.log("[9/10] OPTIMIZER")

        # ------------------------------------------------------
        # Optimizer name
        # ------------------------------------------------------

        optimizer_name = str(
            self.config.optimizer.name
        )

        # ------------------------------------------------------
        # Optimizer parameters
        # ------------------------------------------------------

        optimizer_params = OmegaConf.to_container(
            self.config.optimizer.params,
            resolve=True,
        )

        if optimizer_params is None:
            optimizer_params = {}

        optimizer_params = dict(
            optimizer_params
        )

        # ------------------------------------------------------
        # Build optimizer
        # ------------------------------------------------------

        self.optimizer = build_optimizer(
            model=self.model,
            name=optimizer_name,
            **optimizer_params,
        )

        # ------------------------------------------------------
        # Print information
        # ------------------------------------------------------

        self.log(
            f"       Optimizer: {optimizer_name}"
        )

        self.log(
            f"       Parameters: {optimizer_params}"
        )

        self.log("       OPTIMIZER OK")

    # ======================================================
    # TRAINER
    # ======================================================

    def init_trainer(self) -> None:
        """
        Initialize the trainer.

        Visualization is deliberately NOT passed to the
        trainer. The pipeline handles all plotting.
        """

        self.log("[10/10] TRAINER")

        # --------------------------------------------------
        # Trainer class
        #
        # IMPORTANT:
        #
        # Change "Trainer" below if your actual class is
        # named differently.
        # --------------------------------------------------

        trainer_cls = (
            self.trainer_registry.get(
                "Trainer"
            )
        )

        # --------------------------------------------------
        # Instantiate trainer
        # --------------------------------------------------

        self.trainer = trainer_cls(
            model=self.model,

            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
            test_loader=self.test_loader,

            loss_fn=self.loss_fn,

            optimizer=self.optimizer,

            device=self.device,

            epochs=int(
                self.config.training.epochs
            ),

            patience=int(
                self.config.training.patience
            ),

            gradient_clip=float(
                self.config.training.gradient_clip
            ),

            checkpoint_path=str(
                self.checkpoint_path
            ),

            history_path=str(
                self.history_path
            ),

            results_path=str(
                self.results_path
            ),

            parameter_metrics=(
                self.parameter_metrics
            ),

            trajectory_metrics=(
                self.trajectory_metrics
            ),

            jgd_metrics=(
                self.jgd_metrics
            ),
            run_dir=str(self.run_dir),
            config=OmegaConf.to_container(self.config, resolve=True),
            logger=self.logger,
        )

        self.log("       TRAINER OK")

    # ======================================================
    # INITIALIZE PIPELINE
    # ======================================================

    def init_pipeline(self) -> None:
        """
        Initialize all components in dependency order.
        """

        self.init_data()

        # EDA is optional and independent from training.
        self.run_eda()

        self.init_encoder()

        self.init_ode()

        self.init_solver()

        self.init_model()

        self.init_loss()

        self.init_metrics()

        self.init_optimizer()

        self.init_trainer()

    # ======================================================
    # TRAINING
    # ======================================================

    def fit(self) -> Dict[str, Any]:
        """
        Train the model and create training-history plots.
        """

        self.log("\n")
        self.log("=" * 70)
        self.log("TRAINING")
        self.log("=" * 70)

        history = self.trainer.fit()

        # --------------------------------------------------
        # Training history visualization
        # --------------------------------------------------

        try:

            plot_training_history(
                history=history,
                output_dir=str(
                    self.figure_dir
                ),
            )

            self.log(
                "[INFO] Training history plots saved."
            )

        except Exception as exc:

            self.log(
                "[WARNING] Could not create "
                "training history plots:"
            )

            self.log(
                f"          {exc}"
            )
        self.upload_figures_to_wandb()

        return history

    # ======================================================
    # TEST / EVALUATION
    # ======================================================

    def test(self) -> Dict[str, Any]:
        """
        Evaluate on the complete test set and create
        parameter-recovery and trajectory visualizations.
        """

        self.log("\n")
        self.log("=" * 70)
        self.log("TESTING")
        self.log("=" * 70)

        results = self.trainer.test()
        # Persist complete test output and scalar metrics for run comparison.
        try:
            torch.save(results, self.results_path)
        except Exception as exc:
            self.log("[WARNING] Could not save test results:", exc)
        if isinstance(results, dict):
            self.log_metrics(results, prefix="test")
            summary = {
                "run_name": self.run_dir.name,
                "log_filename": "log.txt",
                "config": OmegaConf.to_container(self.config, resolve=True),
                "metrics": results,
            }
            (self.run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, default=lambda x: x.detach(
                ).cpu().tolist() if torch.is_tensor(x) else str(x)),
                encoding="utf-8",
            )

        # --------------------------------------------------
        # Parameter recovery
        # --------------------------------------------------

        try:

            parameter_names = list(
                self.config.ode.parameter_names
            )

            theta_true = results.get(
                "theta_true"
            )

            theta_pred = results.get(
                "theta_pred"
            )

            if (
                theta_true is not None
                and theta_pred is not None
            ):

                parameter_output_dir = (
                    self.figure_dir
                    / "parameters"
                )

                parameter_output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                plot_parameter_recovery(
                    theta_true=theta_true,
                    theta_pred=theta_pred,
                    parameter_names=parameter_names,
                    output_dir=str(
                        parameter_output_dir
                    ),
                )

                self.log(
                    "[INFO] Parameter recovery "
                    "plots saved."
                )

        except Exception as exc:

            self.log(
                "[WARNING] Could not create "
                "parameter recovery plots:"
            )

            self.log(
                f"          {exc}"
            )

        # --------------------------------------------------
        # Trajectory visualization
        # --------------------------------------------------

        try:

            t = results.get("t")

            trajectory_true = results.get(
                "trajectory_true"
            )

            trajectory_pred = results.get(
                "trajectory_pred"
            )

            trajectory_mask = results.get(
                "trajectory_mask"
            )

            if (
                t is not None
                and trajectory_true is not None
                and trajectory_pred is not None
            ):

                trajectory_output_dir = (
                    self.figure_dir
                    / "trajectories"
                )

                trajectory_output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                state_names = list(
                    self.config.ode.state_names
                )

                num_examples = int(
                    self.config.output.get(
                        "num_trajectory_examples",
                        5,
                    )
                )

                plot_trajectory_examples(
                    t=t,
                    trajectory_true=trajectory_true,
                    trajectory_pred=trajectory_pred,
                    trajectory_mask=trajectory_mask,
                    state_names=state_names,
                    output_dir=str(
                        trajectory_output_dir
                    ),
                    num_examples=num_examples,
                )

                self.log(
                    "[INFO] Trajectory plots saved."
                )

        except Exception as exc:

            self.log(
                "[WARNING] Could not create "
                "trajectory plots:"
            )

            self.log(
                f"          {exc}"
            )

        self.upload_figures_to_wandb()

        return results

    # ======================================================
    # RUN
    # ======================================================

    def run(self) -> Dict[str, Any]:
        """
        Complete experiment:

            1. initialize
            2. train
            3. evaluate
        """

        self.init_pipeline()

        history = self.fit()

        results = self.test()

        self.upload_figures_to_wandb()

        self.log("Experiment completed.")

        return {
            "history": history,
            "results": results,
        }

    # ======================================================
    # REGISTRY DISPLAY
    # ======================================================

    def show_registries(self) -> None:
        """
        Print registered components for debugging.
        """

        self.log("\n")
        self.log("=" * 70)
        self.log("REGISTRIES")
        self.log("=" * 70)

        self.log("\nDATASET")
        self.log(self.dataset_registry)

        self.log("\nENCODER")
        self.log(self.encoder_registry)

        self.log("\nLOSS")
        self.log(self.loss_registry)

        self.log("\nMETRIC")
        self.log(self.metric_registry)

        self.log("\nMODEL")
        self.log(self.model_registry)

        self.log("\nODE")
        self.log(self.ode_registry)

        self.log("\nSOLVER")
        self.log(self.solver_registry)

        self.log("\nTRAINER")
        self.log(self.trainer_registry)


# ==========================================================
# HYDRA ENTRY POINT
# ==========================================================


def log_message(*values, level=None):
    """Module-level logging for the Hydra entry point."""
    logger = LoggerObserver.getLogger("main")
    logger.text(" ".join(str(value) for value in values),
                level=LoggerObserver.INFO if level is None else level)


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config",
)
def main(config: DictConfig) -> None:
    """
    Hydra entry point.
    """

    log_message("=" * 70)
    log_message("PHYSIOME ODE PARAMETER ESTIMATION")
    log_message("=" * 70)

    # ------------------------------------------------------
    # Print current working directory
    # ------------------------------------------------------

    log_message(
        f"Working directory: {Path.cwd()}"
    )

    # ------------------------------------------------------
    # Print selected Hydra components
    # ------------------------------------------------------

    log_message(
        f"Encoder: {config.encoder.name}"
    )

    log_message(
        f"ODE:     {config.ode.name}"
    )

    log_message(
        f"Solver:  {config.solver.name}"
    )

    log_message(
        f"Device:  {config.experiment.device}"
    )

    log_message("=" * 70)

    # ------------------------------------------------------
    # Build and run pipeline
    # ------------------------------------------------------

    pipeline = BasePipeline(config)

    pipeline.run()


# ==========================================================
# SCRIPT ENTRY
# ==========================================================

if __name__ == "__main__":
    main()

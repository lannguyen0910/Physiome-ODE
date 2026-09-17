# ==========================================================
# pipeline.py
# ==========================================================

from pathlib import Path
from typing import Any, Dict, Optional

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

        print("\n" + "=" * 70)
        print("CONFIGURATION")
        print("=" * 70)
        print(OmegaConf.to_yaml(self.config))

    # ======================================================
    # GLOBAL CONFIGURATION
    # ======================================================

    def init_globals(self) -> None:
        """
        Initialize global settings such as random seed,
        device and output directories.
        """

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
            print(
                "[WARNING] CUDA requested but not available. "
                "Falling back to CPU."
            )

            self.device = torch.device("cpu")

        else:
            self.device = torch.device(requested_device)

        print(f"[INFO] Seed   : {seed}")
        print(f"[INFO] Device : {self.device}")

        # --------------------------------------------------
        # Output directories
        # --------------------------------------------------

        self.checkpoint_path = Path(
            to_absolute_path(
                self.config.output.checkpoint
            )
        )

        self.results_path = Path(
            to_absolute_path(
                self.config.output.results
            )
        )

        self.history_path = Path(
            to_absolute_path(
                self.config.output.history
            )
        )

        self.figure_dir = Path(
            to_absolute_path(
                self.config.output.figures
            )
        )

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

        print("[1/10] DATA")

        # --------------------------------------------------
        # Read name from config
        # --------------------------------------------------

        datamodule_name = str(
            self.config.data.name
        )

        print(
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

        print(
            f"       Data path : {data_path}"
        )

        print(
            f"       Batch size: "
            f"{self.config.data.batch_size}"
        )

        print("       DATA OK")

    # ======================================================
    # EDA
    # ======================================================

    def run_eda(self) -> None:
        """
        Run dataset exploratory data analysis.

        Visualization/EDA logic stays outside the trainer.
        """

        print("[2/10] EDA")

        try:
            from datasets.eda import DatasetEDA

        except ImportError as exc:
            print(
                "[WARNING] DatasetEDA could not be imported:"
            )
            print(f"          {exc}")
            print("          Skipping EDA.")
            return

        eda_output_dir = (
            self.figure_dir / "eda"
        )

        eda = DatasetEDA(
            output_dir=str(eda_output_dir)
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

        print("       EDA OK")

    # ======================================================
    # ENCODER
    # ======================================================

    def init_encoder(self) -> None:
        """
        Initialize the selected parameter encoder.
        """

        print("[3/10] ENCODER")

        encoder_name = str(
            self.config.encoder.name
        )

        print(
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

        print(
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

        print("       ENCODER OK")

    # ======================================================
    # ODE
    # ======================================================

    def init_ode(self) -> None:
        """
        Initialize the configured ODE model.
        """

        print("[4/10] ODE")

        ode_name = str(
            self.config.ode.name
        )

        print(
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

        print("       ODE OK")

    # ======================================================
    # SOLVER
    # ======================================================

    def init_solver(self) -> None:
        """
        Initialize the configured ODE solver.
        """

        print("[5/10] SOLVER")

        solver_name = str(
            self.config.solver.name
        )

        print(
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

        print("       SOLVER OK")

    # ======================================================
    # MODEL
    # ======================================================

    def init_model(self) -> None:
        """
        Build the high-level ParameterEstimator by
        composing encoder + ODE + solver.
        """

        print("[6/10] MODEL")

        model_name = str(
            self.config.model.name
        )

        print(
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

        print(
            f"       Parameters: "
            f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}"
        )

        print("       MODEL OK")

    # ======================================================
    # LOSS
    # ======================================================

    def init_loss(self) -> None:
        """
        Initialize the configured loss function.
        """

        print("[7/10] LOSS")

        loss_name = str(
            self.config.loss.name
        )

        print(
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

        print("       LOSS OK")

    # ======================================================
    # METRICS
    # ======================================================

    def init_metrics(self) -> None:
        """
        Initialize parameter, trajectory and JGD metrics.
        """

        print("[8/10] METRICS")

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

        print("       METRICS OK")

    # ======================================================
    # OPTIMIZER
    # ======================================================

    def init_optimizer(self) -> None:
        """
        Initialize the optimizer directly from configuration.

        There is intentionally no optimizer registry.
        """

        print("[9/10] OPTIMIZER")

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

        print(
            f"       Optimizer: {optimizer_name}"
        )

        print(
            f"       Parameters: {optimizer_params}"
        )

        print("       OPTIMIZER OK")

    # ======================================================
    # TRAINER
    # ======================================================

    def init_trainer(self) -> None:
        """
        Initialize the trainer.

        Visualization is deliberately NOT passed to the
        trainer. The pipeline handles all plotting.
        """

        print("[10/10] TRAINER")

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
        )

        print("       TRAINER OK")

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

        print("\n")
        print("=" * 70)
        print("TRAINING")
        print("=" * 70)

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

            print(
                "[INFO] Training history plots saved."
            )

        except Exception as exc:

            print(
                "[WARNING] Could not create "
                "training history plots:"
            )

            print(
                f"          {exc}"
            )

        return history

    # ======================================================
    # TEST / EVALUATION
    # ======================================================

    def test(self) -> Dict[str, Any]:
        """
        Evaluate on the complete test set and create
        parameter-recovery and trajectory visualizations.
        """

        print("\n")
        print("=" * 70)
        print("TESTING")
        print("=" * 70)

        results = self.trainer.test()

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

                print(
                    "[INFO] Parameter recovery "
                    "plots saved."
                )

        except Exception as exc:

            print(
                "[WARNING] Could not create "
                "parameter recovery plots:"
            )

            print(
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

                print(
                    "[INFO] Trajectory plots saved."
                )

        except Exception as exc:

            print(
                "[WARNING] Could not create "
                "trajectory plots:"
            )

            print(
                f"          {exc}"
            )

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

        print("\n")
        print("=" * 70)
        print("REGISTRIES")
        print("=" * 70)

        print("\nDATASET")
        print(self.dataset_registry)

        print("\nENCODER")
        print(self.encoder_registry)

        print("\nLOSS")
        print(self.loss_registry)

        print("\nMETRIC")
        print(self.metric_registry)

        print("\nMODEL")
        print(self.model_registry)

        print("\nODE")
        print(self.ode_registry)

        print("\nSOLVER")
        print(self.solver_registry)

        print("\nTRAINER")
        print(self.trainer_registry)


# ==========================================================
# HYDRA ENTRY POINT
# ==========================================================


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config",
)
def main(config: DictConfig) -> None:
    """
    Hydra entry point.
    """

    print("=" * 70)
    print("PHYSIOME ODE PARAMETER ESTIMATION")
    print("=" * 70)

    # ------------------------------------------------------
    # Print current working directory
    # ------------------------------------------------------

    print(
        f"Working directory: {Path.cwd()}"
    )

    # ------------------------------------------------------
    # Print selected Hydra components
    # ------------------------------------------------------

    print(
        f"Encoder: {config.encoder.name}"
    )

    print(
        f"ODE:     {config.ode.name}"
    )

    print(
        f"Solver:  {config.solver.name}"
    )

    print(
        f"Device:  {config.experiment.device}"
    )

    print("=" * 70)

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

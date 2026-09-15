import os

import hydra
import torch

from hydra.utils import to_absolute_path

from omegaconf import (
    DictConfig,
    OmegaConf,
)


# ==========================================================
# DATASET
# ==========================================================

from datasets import (
    DATAMODULE_REGISTRY,
    DatasetEDA,
)


# ==========================================================
# ENCODERS
# ==========================================================

from encoders import (
    ENCODER_REGISTRY,
)


# ==========================================================
# LOSSES
# ==========================================================

from losses import (
    LOSS_REGISTRY,
)


# ==========================================================
# METRICS
# ==========================================================

from metrics import (
    METRIC_REGISTRY,
)


# ==========================================================
# MODELS
# ==========================================================

from models import (
    MODEL_REGISTRY,
)


# ==========================================================
# ODES
# ==========================================================

from odes import (
    ODE_REGISTRY,
)


# ==========================================================
# SOLVERS
# ==========================================================

from solvers import (
    SOLVER_REGISTRY,
)


# ==========================================================
# TRAINER
# ==========================================================

from trainer import (
    TRAINER_REGISTRY,
)


# ==========================================================
# UTILITIES
# ==========================================================

from utilities import (
    seed_everything,
    plot_parameter_recovery,
    plot_training_history,
    plot_trajectory_examples,
)


class BasePipeline:

    def __init__(
        self,
        config: DictConfig,
    ):

        self.config = config

        self.initialized = False

        # --------------------------------------------------
        # Reproducibility
        # --------------------------------------------------

        seed_everything(
            int(
                config.experiment.seed
            )
        )

        # --------------------------------------------------
        # Global initialization
        # --------------------------------------------------

        self.init_globals()

        self.init_registry()

    # ======================================================
    # GLOBALS
    # ======================================================

    def init_globals(self):

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        requested_device = str(
            self.config.experiment.device
        )

        if (
            requested_device == "cuda"
            and not torch.cuda.is_available()
        ):

            print(
                "WARNING: CUDA requested "
                "but is not available."
            )

            print(
                "Falling back to CPU."
            )

            self.device = torch.device(
                "cpu"
            )

        else:

            self.device = torch.device(
                requested_device
            )

        # --------------------------------------------------
        # Output paths
        # --------------------------------------------------

        self.checkpoint_path = (
            to_absolute_path(
                self.config.output.checkpoint
            )
        )

        self.results_path = (
            to_absolute_path(
                self.config.output.results
            )
        )

        self.history_path = (
            to_absolute_path(
                self.config.output.history
            )
        )

        self.figure_dir = (
            to_absolute_path(
                self.config.output.figures
            )
        )

        # --------------------------------------------------
        # Create directories
        # --------------------------------------------------

        os.makedirs(
            self.figure_dir,
            exist_ok=True,
        )

        checkpoint_dir = (
            os.path.dirname(
                self.checkpoint_path
            )
        )

        results_dir = (
            os.path.dirname(
                self.results_path
            )
        )

        history_dir = (
            os.path.dirname(
                self.history_path
            )
        )

        if checkpoint_dir:

            os.makedirs(
                checkpoint_dir,
                exist_ok=True,
            )

        if results_dir:

            os.makedirs(
                results_dir,
                exist_ok=True,
            )

        if history_dir:

            os.makedirs(
                history_dir,
                exist_ok=True,
            )

        # --------------------------------------------------
        # Print config
        # --------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "PHYSIOME ODE PIPELINE"
        )

        print(
            "=" * 70
        )

        print(
            f"Device: {self.device}"
        )

        print(
            f"Seed: "
            f"{self.config.experiment.seed}"
        )

        print(
            "\nFinal configuration:"
        )

        print(
            OmegaConf.to_yaml(
                self.config
            )
        )

    # ======================================================
    # REGISTRIES
    # ======================================================

    def init_registry(self):

        self.datamodule_registry = (
            DATAMODULE_REGISTRY
        )

        self.encoder_registry = (
            ENCODER_REGISTRY
        )

        self.ode_registry = (
            ODE_REGISTRY
        )

        self.solver_registry = (
            SOLVER_REGISTRY
        )

        self.model_registry = (
            MODEL_REGISTRY
        )

        self.loss_registry = (
            LOSS_REGISTRY
        )

        self.metric_registry = (
            METRIC_REGISTRY
        )

        self.trainer_registry = (
            TRAINER_REGISTRY
        )

    # ======================================================
    # DATA
    # ======================================================

    def init_data(self):

        print(
            "\n[1/10] DATA"
        )

        data_config = (
            self.config.data
        )

        datamodule_name = (
            data_config.get(
                "name",
                "physiome",
            )
        )

        datamodule_cls = (
            self.datamodule_registry.get(
                datamodule_name
            )
        )

        data_path = (
            to_absolute_path(
                data_config.path
            )
        )

        self.datamodule = datamodule_cls(
            data_path=data_path,

            batch_size=int(
                data_config.batch_size
            ),

            num_workers=int(
                data_config.num_workers
            ),
        )

        self.datamodule.setup()

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
            f"Data module: "
            f"{datamodule_name}"
        )

        print(
            f"Data path: "
            f"{data_path}"
        )

    # ======================================================
    # DATASET EDA
    # ======================================================

    def run_eda(self):

        print(
            "\n[2/10] DATASET EDA"
        )

        eda_output_dir = (
            os.path.join(
                self.figure_dir,
                "eda",
            )
        )

        os.makedirs(
            eda_output_dir,
            exist_ok=True,
        )

        eda = DatasetEDA(
            output_dir=eda_output_dir
        )

        # --------------------------------------------------
        # Train
        # --------------------------------------------------

        eda.analyze(
            dataset=(
                self.datamodule.train_dataset
            ),
            dataset_name="train",
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        eda.analyze(
            dataset=(
                self.datamodule.valid_dataset
            ),
            dataset_name="validation",
        )

        # --------------------------------------------------
        # Test
        # --------------------------------------------------

        eda.analyze(
            dataset=(
                self.datamodule.test_dataset
            ),
            dataset_name="test",
        )

        print(
            f"EDA saved to: "
            f"{eda_output_dir}"
        )

    # ======================================================
    # ENCODER
    # ======================================================

    def init_encoder(self):

        print(
            "\n[3/10] ENCODER"
        )

        encoder_config = (
            self.config.encoder
        )

        encoder_name = (
            encoder_config.name
        )

        encoder_cls = (
            self.encoder_registry.get(
                encoder_name
            )
        )

        params = dict(
            encoder_config.get(
                "params",
                {},
            )
        )

        state_dim = int(
            self.config.model.state_dim
        )

        parameter_dim = int(
            self.config.model.parameter_dim
        )

        # Input:
        #
        #   time        -> 1
        #   state       -> D
        #   mask        -> D
        #
        # total = 1 + D + D

        input_dim = (
            1
            + state_dim
            + state_dim
        )

        params.update(
            {
                "input_dim": input_dim,
                "state_dim": state_dim,
                "parameter_dim": parameter_dim,
            }
        )

        self.encoder = encoder_cls(
            **params
        )

        print(
            f"Encoder: "
            f"{encoder_name}"
        )

        print(
            f"Input dimension: "
            f"{input_dim}"
        )

        print(
            "Trainable parameters: "
            f"{sum(p.numel() for p in self.encoder.parameters()):,}"
        )

    # ======================================================
    # ODE
    # ======================================================

    def init_ode(self):

        print(
            "\n[4/10] ODE"
        )

        ode_config = (
            self.config.ode
        )

        ode_name = (
            ode_config.name
        )

        ode_cls = (
            self.ode_registry.get(
                ode_name
            )
        )

        self.ode = ode_cls(

            parameters=dict(
                ode_config.get(
                    "parameters",
                    {},
                )
            ),

            initial_state=dict(
                ode_config.get(
                    "initial_state",
                    {},
                )
            ),

            parameter_names=list(
                ode_config.get(
                    "parameter_names",
                    [],
                )
            ),

            state_names=list(
                ode_config.get(
                    "state_names",
                    [],
                )
            ),
        )

        print(
            f"ODE: "
            f"{ode_name}"
        )

    # ======================================================
    # SOLVER
    # ======================================================

    def init_solver(self):

        print(
            "\n[5/10] SOLVER"
        )

        solver_config = (
            self.config.solver
        )

        solver_name = (
            solver_config.name
        )

        solver_cls = (
            self.solver_registry.get(
                solver_name
            )
        )

        self.solver = solver_cls(
            **dict(
                solver_config.get(
                    "params",
                    {},
                )
            )
        )

        print(
            f"Solver: "
            f"{solver_name}"
        )

    # ======================================================
    # MODEL
    # ======================================================

    def init_model(self):

        print(
            "\n[6/10] MODEL"
        )

        model_config = (
            self.config.model
        )

        model_name = (
            model_config.name
        )

        model_cls = (
            self.model_registry.get(
                model_name
            )
        )

        self.model = model_cls(

            encoder=self.encoder,

            ode=self.ode,

            solver=self.solver,

            # The predicted y0 corresponds to t=0.
            initial_state_time=0.0,
        )

        print(
            f"Model: "
            f"{model_name}"
        )

    # ======================================================
    # LOSS
    # ======================================================

    def init_loss(self):

        print(
            "\n[7/10] LOSS"
        )

        loss_config = (
            self.config.loss
        )

        loss_name = (
            loss_config.get(
                "name",
                "parameter_estimation",
            )
        )

        loss_cls = (
            self.loss_registry.get(
                loss_name
            )
        )

        self.loss_fn = loss_cls(

            parameter_loss_weight=float(
                self.config.training
                .parameter_loss_weight
            ),

            trajectory_loss_weight=float(
                self.config.training
                .trajectory_loss_weight
            ),
        )

        print(
            f"Loss: "
            f"{loss_name}"
        )

    # ======================================================
    # METRICS
    # ======================================================

    def init_metrics(self):

        print(
            "\n[8/10] METRICS"
        )

        self.parameter_metrics = None

        self.trajectory_metrics = None

        self.jgd_metrics = None

        # --------------------------------------------------
        # Parameter metrics
        # --------------------------------------------------

        parameter_config = (
            self.config.metrics.parameter
        )

        if parameter_config.enabled:

            metric_cls = (
                self.metric_registry.get(
                    "parameter"
                )
            )

            self.parameter_metrics = (
                metric_cls(
                    methods=list(
                        parameter_config.methods
                    ),

                    **dict(
                        parameter_config.get(
                            "params",
                            {},
                        )
                    ),
                )
            )

        # --------------------------------------------------
        # Trajectory metrics
        # --------------------------------------------------

        trajectory_config = (
            self.config.metrics.trajectory
        )

        if trajectory_config.enabled:

            metric_cls = (
                self.metric_registry.get(
                    "trajectory"
                )
            )

            self.trajectory_metrics = (
                metric_cls(
                    methods=list(
                        trajectory_config.methods
                    ),

                    **dict(
                        trajectory_config.get(
                            "params",
                            {},
                        )
                    ),
                )
            )

        # --------------------------------------------------
        # JGD
        # --------------------------------------------------

        jgd_config = (
            self.config.metrics.jgd
        )

        if jgd_config.enabled:

            metric_cls = (
                self.metric_registry.get(
                    "jgd"
                )
            )

            self.jgd_metrics = (
                metric_cls(
                    methods=list(
                        jgd_config.methods
                    ),

                    **dict(
                        jgd_config.get(
                            "params",
                            {},
                        )
                    ),
                )
            )

    # ======================================================
    # OPTIMIZER
    # ======================================================

    def init_optimizer(self):

        print(
            "\n[9/10] OPTIMIZER"
        )

        optimizer_config = (
            self.config.optimizer
        )

        optimizer_name = (
            optimizer_config.name
        )

        if optimizer_name.lower() == "adamw":

            self.optimizer = (
                torch.optim.AdamW(
                    self.model.parameters(),

                    lr=float(
                        optimizer_config.params
                        .learning_rate
                    ),

                    weight_decay=float(
                        optimizer_config.params
                        .weight_decay
                    ),
                )
            )

        elif optimizer_name.lower() == "adam":

            self.optimizer = (
                torch.optim.Adam(
                    self.model.parameters(),

                    lr=float(
                        optimizer_config.params
                        .learning_rate
                    ),

                    weight_decay=float(
                        optimizer_config.params
                        .weight_decay
                    ),
                )
            )

        elif optimizer_name.lower() == "sgd":

            self.optimizer = (
                torch.optim.SGD(
                    self.model.parameters(),

                    lr=float(
                        optimizer_config.params
                        .learning_rate
                    ),

                    weight_decay=float(
                        optimizer_config.params
                        .weight_decay
                    ),
                )
            )

        else:

            raise ValueError(
                f"Unknown optimizer: "
                f"{optimizer_name}"
            )

        print(
            f"Optimizer: "
            f"{optimizer_name}"
        )

    # ======================================================
    # TRAINER
    # ======================================================

    def init_trainer(self):

        print(
            "\n[10/10] TRAINER"
        )

        trainer_cls = (
            self.trainer_registry.get(
                "default"
            )
        )

        training_config = (
            self.config.training
        )

        self.trainer = trainer_cls(

            model=self.model,

            train_loader=self.train_loader,

            valid_loader=self.valid_loader,

            test_loader=self.test_loader,

            loss_fn=self.loss_fn,

            optimizer=self.optimizer,

            device=self.device,

            epochs=int(
                training_config.epochs
            ),

            patience=int(
                training_config.patience
            ),

            gradient_clip=float(
                training_config.gradient_clip
            ),

            checkpoint_path=(
                self.checkpoint_path
            ),

            history_path=(
                self.history_path
            ),

            results_path=(
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

    # ======================================================
    # INITIALIZE PIPELINE
    # ======================================================

    def init_pipeline(self):

        if self.initialized:

            return

        # --------------------------------------------------
        # 1. Load data
        # --------------------------------------------------

        self.init_data()

        # --------------------------------------------------
        # 2. EDA BEFORE TRAINING
        # --------------------------------------------------

        self.run_eda()

        # --------------------------------------------------
        # 3. Components
        # --------------------------------------------------

        self.init_encoder()

        self.init_ode()

        self.init_solver()

        self.init_model()

        # --------------------------------------------------
        # 4. Training components
        # --------------------------------------------------

        self.init_loss()

        self.init_metrics()

        self.init_optimizer()

        self.init_trainer()

        self.initialized = True

        print(
            "\n"
            + "=" * 70
        )

        print(
            "PIPELINE INITIALIZED"
        )

        print(
            "=" * 70
        )

    # ======================================================
    # TRAIN
    # ======================================================

    def fit(self):

        self.init_pipeline()

        history = (
            self.trainer.fit()
        )

        # --------------------------------------------------
        # Visualization
        # --------------------------------------------------

        plot_training_history(
            history=history,
            output_dir=self.figure_dir,
        )

        print(
            f"Training figures saved to: "
            f"{self.figure_dir}"
        )

        return history

    # ======================================================
    # TEST
    # ======================================================

    def test(self):

        self.init_pipeline()

        results = (
            self.trainer.test()
        )

        # --------------------------------------------------
        # Output folders
        # --------------------------------------------------

        parameter_output_dir = (
            os.path.join(
                self.figure_dir,
                "parameters",
            )
        )

        trajectory_output_dir = (
            os.path.join(
                self.figure_dir,
                "trajectories",
            )
        )

        # --------------------------------------------------
        # Parameter recovery
        # --------------------------------------------------

        plot_parameter_recovery(

            theta_true=(
                results[
                    "theta_true"
                ]
            ),

            theta_pred=(
                results[
                    "theta_pred"
                ]
            ),

            parameter_names=list(
                self.config.ode
                .parameter_names
            ),

            output_dir=(
                parameter_output_dir
            ),
        )

        # --------------------------------------------------
        # Trajectory reconstruction
        # --------------------------------------------------

        plot_trajectory_examples(

            t=results[
                "t"
            ],

            trajectory_true=(
                results[
                    "trajectory_true"
                ]
            ),

            trajectory_pred=(
                results[
                    "trajectory_pred"
                ]
            ),

            trajectory_mask=(
                results.get(
                    "trajectory_mask",
                    None,
                )
            ),

            state_names=list(
                self.config.ode
                .state_names
            ),

            output_dir=(
                trajectory_output_dir
            ),

            num_examples=int(
                self.config.output.get(
                    "num_trajectory_examples",
                    5,
                )
            ),
        )

        print(
            "\nTest figures saved:"
        )

        print(
            f"  Parameters: "
            f"{parameter_output_dir}"
        )

        print(
            f"  Trajectories: "
            f"{trajectory_output_dir}"
        )

        return results

    # ======================================================
    # COMPLETE RUN
    # ======================================================

    def run(self):

        self.init_pipeline()

        history = self.fit()

        results = self.test()

        return {
            "history": history,
            "results": results,
        }

    # ======================================================
    # REGISTRIES
    # ======================================================

    def print_registries(self):

        print(
            "\n"
            + "=" * 70
        )

        print(
            "AVAILABLE COMPONENTS"
        )

        print(
            "=" * 70
        )

        print(
            "\nDATA MODULES"
        )

        print(
            self.datamodule_registry
        )

        print(
            "\nENCODERS"
        )

        print(
            self.encoder_registry
        )

        print(
            "\nODES"
        )

        print(
            self.ode_registry
        )

        print(
            "\nSOLVERS"
        )

        print(
            self.solver_registry
        )

        print(
            "\nMODELS"
        )

        print(
            self.model_registry
        )

        print(
            "\nLOSSES"
        )

        print(
            self.loss_registry
        )

        print(
            "\nMETRICS"
        )

        print(
            self.metric_registry
        )

        print(
            "\nTRAINERS"
        )

        print(
            self.trainer_registry
        )


# ==========================================================
# HYDRA ENTRY POINT
# ==========================================================

@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config",
)
def main(
    config: DictConfig,
):

    pipeline = BasePipeline(
        config
    )

    pipeline.run()


if __name__ == "__main__":

    main()

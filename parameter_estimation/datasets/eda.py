import os
import torch
import matplotlib.pyplot as plt
from utilities.loggers import LoggerObserver


class DatasetEDA:
    def __init__(
        self,
        output_dir="figures/eda",
        logger=None,
    ):
        self.output_dir = output_dir
        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )
        self.logger = logger or LoggerObserver.getLogger("main")

    def analyze(
        self,
        dataset,
        dataset_name="train",
    ):

        self.log("\n" + "=" * 40)
        self.log(f"EDA: {dataset_name.upper()} DATASET")
        self.log("=" * 40)

        self.log(f"Number of samples: {len(dataset)}")

        sample = dataset[0]

        if isinstance(sample, dict):
            self.log("\nAvailable keys:")
            self.log(list(sample.keys()))

            self._analyze_dict_dataset(
                dataset=dataset,
                dataset_name=dataset_name,
            )

        else:

            self.log(
                "\nWARNING: Dataset sample is not a dictionary."
            )
            self.log(
                "Please adapt EDA to your dataset format."
            )

        self.log("=" * 40 + "\n")

    def _collect_data(
        self,
        dataset,
        key,
    ):

        values = []

        for i in range(len(dataset)):

            sample = dataset[i]

            if key in sample:

                values.append(
                    sample[key]
                )

        if len(values) == 0:

            return None

        try:

            return torch.stack(values)

        except RuntimeError:

            return values

    def _analyze_dict_dataset(
        self,
        dataset,
        dataset_name,
    ):

        T = self._collect_data(
            dataset,
            "T",
        )

        X = self._collect_data(
            dataset,
            "X",
        )

        M = self._collect_data(
            dataset,
            "M",
        )

        TY = self._collect_data(
            dataset,
            "TY",
        )

        Y = self._collect_data(
            dataset,
            "Y",
        )

        theta = self._collect_data(
            dataset,
            "theta",
        )

        if theta is None:

            theta = self._collect_data(
                dataset,
                "parameters",
            )

        # --------------------------------------------------
        # Basic shapes
        # --------------------------------------------------

        self.log("\n1. DATA SHAPES")
        self.log("-" * 40)

        for name, value in {
            "T": T,
            "X": X,
            "M": M,
            "TY": TY,
            "Y": Y,
            "theta": theta,
        }.items():

            if value is not None and torch.is_tensor(value):

                self.log(
                    f"{name}: {tuple(value.shape)}"
                )

        # --------------------------------------------------
        # Time analysis
        # --------------------------------------------------

        if T is not None:

            self.analyze_time(
                T=T,
                dataset_name=dataset_name,
                time_type="observation",
            )

        if TY is not None:

            self.analyze_time(
                T=TY,
                dataset_name=dataset_name,
                time_type="forecast",
            )

        # --------------------------------------------------
        # State analysis
        # --------------------------------------------------

        if X is not None:

            self.analyze_states(
                states=X,
                name="Observed states X",
                dataset_name=dataset_name,
            )

        if Y is not None:

            self.analyze_states(
                states=Y,
                name="Forecast states Y",
                dataset_name=dataset_name,
            )

        # --------------------------------------------------
        # Missing observations
        # --------------------------------------------------

        if M is not None:

            self.analyze_mask(
                M=M,
                dataset_name=dataset_name,
            )

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------

        if theta is not None:

            self.analyze_parameters(
                theta=theta,
                dataset_name=dataset_name,
            )

        # --------------------------------------------------
        # Example trajectories
        # --------------------------------------------------

        if T is not None and X is not None:

            self.plot_examples(
                T=T,
                X=X,
                dataset_name=dataset_name,
            )

    # ======================================================
    # TIME ANALYSIS
    # ======================================================

    def analyze_time(
        self,
        T,
        dataset_name,
        time_type,
    ):

        self.log(f"\n2. TIME ANALYSIS ({time_type})")
        self.log("-" * 40)

        if T.dim() == 3:

            T = T.squeeze(-1)

        total_duplicates = 0

        all_gaps = []

        for i in range(T.shape[0]):

            t = T[i]

            valid = torch.isfinite(t)

            t = t[valid]

            if len(t) == 0:

                continue

            unique_t = torch.unique(
                t,
                sorted=True,
            )

            duplicates = len(t) - len(unique_t)

            total_duplicates += duplicates

            if len(unique_t) > 1:

                gaps = unique_t[1:] - unique_t[:-1]

                all_gaps.append(gaps)

        self.log(
            f"Total duplicate timestamps: {total_duplicates}"
        )

        if len(all_gaps) > 0:

            all_gaps = torch.cat(all_gaps)

            self.log(
                f"Time gap min: {all_gaps.min().item():.6f}"
            )

            self.log(
                f"Time gap mean: {all_gaps.mean().item():.6f}"
            )

            self.log(
                f"Time gap max: {all_gaps.max().item():.6f}"
            )

            self.plot_time_gaps(
                gaps=all_gaps,
                dataset_name=dataset_name,
                time_type=time_type,
            )

    # ======================================================
    # STATE ANALYSIS
    # ======================================================

    def analyze_states(
        self,
        states,
        name,
        dataset_name,
    ):

        self.log(f"\n3. {name.upper()}")
        self.log("-" * 40)

        state_dim = states.shape[-1]

        for state_idx in range(state_dim):

            values = states[..., state_idx]

            finite = values[
                torch.isfinite(values)
            ]

            if len(finite) == 0:

                continue

            self.log(f"\nState {state_idx}")

            self.log(
                f"Min: {finite.min().item():.6f}"
            )

            self.log(
                f"Max: {finite.max().item():.6f}"
            )

            self.log(
                f"Mean: {finite.mean().item():.6f}"
            )

            self.log(
                f"Std: {finite.std().item():.6f}"
            )

            self.plot_distribution(
                values=finite,
                title=(
                    f"{dataset_name} - "
                    f"{name} - State {state_idx}"
                ),
                filename=(
                    f"{dataset_name}_"
                    f"{name.replace(' ', '_')}_"
                    f"state_{state_idx}.png"
                ),
            )

    # ======================================================
    # MASK ANALYSIS
    # ======================================================

    def analyze_mask(
        self,
        M,
        dataset_name,
    ):

        self.log("\n4. OBSERVATION MASK ANALYSIS")
        self.log("-" * 40)

        total = M.numel()

        observed = M.sum().item()

        observation_ratio = observed / total

        self.log(
            f"Total entries: {total}"
        )

        self.log(
            f"Observed entries: {observed}"
        )

        self.log(
            f"Observation ratio: "
            f"{observation_ratio:.4f}"
        )

        state_dim = M.shape[-1]

        for state_idx in range(state_dim):

            ratio = (
                M[..., state_idx]
                .float()
                .mean()
                .item()
            )

            self.log(
                f"State {state_idx} observation ratio: "
                f"{ratio:.4f}"
            )

    # ======================================================
    # PARAMETER ANALYSIS
    # ======================================================

    def analyze_parameters(
        self,
        theta,
        dataset_name,
    ):

        self.log("\n5. PARAMETER ANALYSIS")
        self.log("-" * 40)

        if theta.dim() == 1:
            theta = theta.unsqueeze(0)

        parameter_dim = theta.shape[-1]

        for param_idx in range(parameter_dim):
            values = theta[:, param_idx]

            finite = values[
                torch.isfinite(values)
            ]

            if len(finite) == 0:

                continue

            self.log(f"\nParameter {param_idx}")

            self.log(
                f"Min: {finite.min().item():.6f}"
            )

            self.log(
                f"Max: {finite.max().item():.6f}"
            )

            self.log(
                f"Mean: {finite.mean().item():.6f}"
            )

            self.log(
                f"Std: {finite.std().item():.6f}"
            )

            self.plot_distribution(
                values=finite,
                title=(
                    f"{dataset_name} - "
                    f"Parameter {param_idx}"
                ),
                filename=(
                    f"{dataset_name}_"
                    f"parameter_{param_idx}.png"
                ),
            )

    # ======================================================
    # DISTRIBUTION PLOT
    # ======================================================

    def plot_distribution(
        self,
        values,
        title,
        filename,
    ):

        values = (
            values
            .detach()
            .cpu()
            .numpy()
        )

        plt.figure(
            figsize=(7, 4),
        )

        plt.hist(
            values,
            bins=50,
        )

        plt.title(title)

        plt.xlabel("Value")

        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_dir,
                filename,
            ),
            dpi=150,
        )

        plt.close()

    # ======================================================
    # TIME GAP PLOT
    # ======================================================

    def plot_time_gaps(
        self,
        gaps,
        dataset_name,
        time_type,
    ):

        gaps = (
            gaps
            .detach()
            .cpu()
            .numpy()
        )

        plt.figure(
            figsize=(7, 4),
        )

        plt.hist(
            gaps,
            bins=50,
        )

        plt.title(
            f"{dataset_name} "
            f"{time_type} time gaps"
        )

        plt.xlabel("Time difference")

        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_dir,
                f"{dataset_name}_{time_type}_time_gaps.png",
            ),
            dpi=150,
        )

        plt.close()

    # ======================================================
    # EXAMPLE TRAJECTORIES
    # ======================================================

    def plot_examples(
        self,
        T,
        X,
        dataset_name,
        num_examples=5,
    ):

        if T.dim() == 3:

            T = T.squeeze(-1)

        num_examples = min(
            num_examples,
            T.shape[0],
        )

        state_dim = X.shape[-1]

        for state_idx in range(state_dim):

            plt.figure(
                figsize=(10, 5),
            )

            for sample_idx in range(num_examples):

                t = T[sample_idx]

                x = X[
                    sample_idx,
                    :,
                    state_idx,
                ]

                valid = (
                    torch.isfinite(t)
                    &
                    torch.isfinite(x)
                )

                t_valid = (
                    t[valid]
                    .detach()
                    .cpu()
                    .numpy()
                )

                x_valid = (
                    x[valid]
                    .detach()
                    .cpu()
                    .numpy()
                )

                plt.plot(
                    t_valid,
                    x_valid,
                    label=f"Sample {sample_idx}",
                )

            plt.title(
                f"{dataset_name} "
                f"observed trajectories - "
                f"State {state_idx}"
            )

            plt.xlabel("Time")

            plt.ylabel("State value")

            plt.legend()

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    self.output_dir,
                    f"{dataset_name}_"
                    f"trajectory_state_{state_idx}.png",
                ),
                dpi=150,
            )

            plt.close()

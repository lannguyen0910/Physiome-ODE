import torch


class JGDMetrics:
    def __init__(
        self,
        eps=1e-8,
        methods=None,
    ):

        self.eps = float(eps)

        if methods is None:

            methods = [
                "pointwise_error",
                "trajectory_distribution_error",
                "temporal_dynamics_error",
                "jgd_score",
            ]

        self.methods = list(
            methods
        )

    # =====================================================
    # POINTWISE ERROR
    # =====================================================

    def pointwise_error(
        self,
        prediction,
        target,
    ):

        absolute_error = torch.abs(
            prediction - target
        )

        state_mae = (
            absolute_error.mean(
                dim=(0, 1)
            )
        )

        state_range = (
            target.amax(
                dim=(0, 1)
            )
            -
            target.amin(
                dim=(0, 1)
            )
        )

        normalized_state_error = (
            state_mae
            /
            (
                state_range
                + self.eps
            )
        )

        return (
            normalized_state_error.mean()
        )

    # =====================================================
    # DISTRIBUTION ERROR
    # =====================================================

    def trajectory_distribution_error(
        self,
        prediction,
        target,
    ):

        pred_mean = (
            prediction.mean(
                dim=(0, 1)
            )
        )

        target_mean = (
            target.mean(
                dim=(0, 1)
            )
        )

        pred_std = (
            prediction.std(
                dim=(0, 1)
            )
        )

        target_std = (
            target.std(
                dim=(0, 1)
            )
        )

        state_range = (
            target.amax(
                dim=(0, 1)
            )
            -
            target.amin(
                dim=(0, 1)
            )
        )

        mean_error = torch.mean(
            torch.abs(
                pred_mean
                -
                target_mean
            )
            /
            (
                state_range
                + self.eps
            )
        )

        std_error = torch.mean(
            torch.abs(
                pred_std
                -
                target_std
            )
            /
            (
                state_range
                + self.eps
            )
        )

        return (
            mean_error
            +
            std_error
        )

    # =====================================================
    # TEMPORAL DYNAMICS
    # =====================================================

    def temporal_dynamics_error(
        self,
        prediction,
        target,
    ):

        pred_diff = (
            prediction[:, 1:]
            -
            prediction[:, :-1]
        )

        target_diff = (
            target[:, 1:]
            -
            target[:, :-1]
        )

        numerator = torch.mean(
            torch.abs(
                pred_diff
                -
                target_diff
            )
        )

        denominator = torch.mean(
            torch.abs(
                target_diff
            )
        )

        return (
            numerator
            /
            (
                denominator
                + self.eps
            )
        )

    # =====================================================
    # COMBINED SCORE
    # =====================================================

    def jgd_score(
        self,
        prediction,
        target,
    ):

        point_error = (
            self.pointwise_error(
                prediction,
                target,
            )
        )

        distribution_error = (
            self.trajectory_distribution_error(
                prediction,
                target,
            )
        )

        dynamics_error = (
            self.temporal_dynamics_error(
                prediction,
                target,
            )
        )

        return (
            point_error
            +
            distribution_error
            +
            dynamics_error
        ) / 3.0

    # =====================================================
    # COMPUTE
    # =====================================================

    def compute(
        self,
        prediction,
        target,
    ):

        results = {}

        for method_name in self.methods:

            method = getattr(
                self,
                method_name,
            )

            results[method_name] = method(
                prediction,
                target,
            )

        return results

    def __call__(
        self,
        prediction,
        target,
    ):

        return self.compute(
            prediction,
            target,
        )

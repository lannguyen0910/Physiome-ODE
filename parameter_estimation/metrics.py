import torch


class ParameterMetrics:

    @staticmethod
    def mae(prediction, target):

        return torch.mean(
            torch.abs(prediction - target)
        )

    @staticmethod
    def rmse(prediction, target):

        return torch.sqrt(
            torch.mean(
                (prediction - target) ** 2
            )
        )

    @staticmethod
    def relative_error(
        prediction,
        target,
        eps=1e-8,
    ):
        """
        Mean Relative Error:

        |pred - true|
        -------------
        |true| + eps
        """

        relative_error = (
            torch.abs(prediction - target)
            /
            (torch.abs(target) + eps)
        )

        return torch.mean(relative_error)

    @staticmethod
    def normalized_error(
        prediction,
        target,
        parameter_scale,
        eps=1e-8,
    ):
        """
        Normalized Parameter Error.
        """

        error = torch.abs(prediction - target)

        normalized_error = (
            error
            /
            (parameter_scale + eps)
        )

        return torch.mean(normalized_error)

    @staticmethod
    def per_parameter_relative_error(
        prediction,
        target,
        eps=1e-8,
    ):
        """
        Returns one relative error for each parameter.

        Input:
            prediction: [B, P]
            target:     [B, P]

        Output:
            [P]
        """

        relative_error = (
            torch.abs(prediction - target)
            /
            (torch.abs(target) + eps)
        )

        return torch.mean(
            relative_error,
            dim=0,
        )

    @staticmethod
    def per_parameter_accuracy(
        prediction,
        target,
        tolerance=0.1,
        eps=1e-8,
    ):
        """
        Parameter accuracy based on relative error.

        A prediction is considered correct if:

        |pred - true|
        -------------
        |true| + eps

        <= tolerance

        tolerance = 0.1 means within 10%.
        """

        relative_error = (
            torch.abs(prediction - target)
            /
            (torch.abs(target) + eps)
        )

        correct = (
            relative_error <= tolerance
        ).float()

        return torch.mean(
            correct,
            dim=0,
        )

    @staticmethod
    def overall_parameter_accuracy(
        prediction,
        target,
        tolerance=0.1,
        eps=1e-8,
    ):
        """
        Average accuracy across all parameters.
        """

        per_parameter_accuracy = (
            ParameterMetrics.per_parameter_accuracy(
                prediction,
                target,
                tolerance=tolerance,
                eps=eps,
            )
        )

        return torch.mean(
            per_parameter_accuracy
        )

    @staticmethod
    def log_mae(
        prediction,
        target,
        eps=1e-8,
    ):
        """
        Useful for strictly positive parameters.
        """

        prediction = torch.clamp(
            prediction,
            min=eps,
        )

        target = torch.clamp(
            target,
            min=eps,
        )

        return torch.mean(
            torch.abs(
                torch.log(prediction)
                -
                torch.log(target)
            )
        )


class TrajectoryMetrics:

    @staticmethod
    def mae(prediction, target):

        return torch.mean(
            torch.abs(prediction - target)
        )

    @staticmethod
    def rmse(prediction, target):

        return torch.sqrt(
            torch.mean(
                (prediction - target) ** 2
            )
        )

    @staticmethod
    def normalized_rmse(
        prediction,
        target,
        eps=1e-8,
    ):
        """
        RMSE normalized by trajectory scale.
        """

        rmse = TrajectoryMetrics.rmse(
            prediction,
            target,
        )

        scale = (
            target.max()
            -
            target.min()
        )

        return rmse / (scale + eps)


class JGDMetrics:
    @staticmethod
    def pointwise_error(
        prediction,
        target,
        eps=1e-8,
    ):
        """
        Stable point-wise trajectory error.

        Instead of dividing every error by the
        corresponding target value, normalize each
        state by its overall trajectory range.

        prediction:
            [B, T, D]

        target:
            [B, T, D]

        Returns:
            Scalar normalized point-wise error.
        """

        absolute_error = torch.abs(
            prediction - target
        )

        # MAE for each state variable
        state_mae = absolute_error.mean(
            dim=(0, 1)
        )

        # Range of each state variable
        state_range = (
            target.amax(dim=(0, 1))
            -
            target.amin(dim=(0, 1))
        )

        normalized_state_error = (
            state_mae
            /
            (state_range + eps)
        )

        return normalized_state_error.mean()

    @staticmethod
    def trajectory_distribution_error(
        prediction,
        target,
        eps=1e-8,
    ):
        """
        JGD-inspired distribution comparison.

        Compare mean and standard deviation
        of predicted and true trajectories.
        """

        pred_mean = prediction.mean(
            dim=(0, 1)
        )

        target_mean = target.mean(
            dim=(0, 1)
        )

        pred_std = prediction.std(
            dim=(0, 1)
        )

        target_std = target.std(
            dim=(0, 1)
        )

        state_range = (
            target.amax(dim=(0, 1))
            -
            target.amin(dim=(0, 1))
        )

        mean_error = torch.mean(
            torch.abs(
                pred_mean - target_mean
            )
            /
            (state_range + eps)
        )

        std_error = torch.mean(
            torch.abs(
                pred_std - target_std
            )
            /
            (state_range + eps)
        )

        return mean_error + std_error

    @staticmethod
    def temporal_dynamics_error(
        prediction,
        target,
        eps=1e-8,
    ):
        """
        Compare temporal changes.
        """

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

        # More numerically stable than dividing
        # every tiny temporal difference directly.
        numerator = torch.mean(
            torch.abs(
                pred_diff - target_diff
            )
        )

        denominator = torch.mean(
            torch.abs(target_diff)
        )

        return numerator / (
            denominator + eps
        )

    @staticmethod
    def jgd_score(
        prediction,
        target,
        eps=1e-8,
    ):
        """
        JGD-inspired combined trajectory score.

        Lower is better.

        Components:
        1. Point-wise normalized error
        2. Distribution error
        3. Temporal dynamics error
        """

        point_error = JGDMetrics.pointwise_error(
            prediction,
            target,
            eps,
        )

        distribution_error = JGDMetrics.trajectory_distribution_error(
            prediction,
            target,
            eps,
        )

        dynamics_error = JGDMetrics.temporal_dynamics_error(
            prediction,
            target,
            eps,
        )

        score = (
            point_error
            +
            distribution_error
            +
            dynamics_error
        ) / 3.0

        return score

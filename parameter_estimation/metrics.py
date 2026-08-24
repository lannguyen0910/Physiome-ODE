import torch


class ParameterMetrics:
    @staticmethod
    def mae(
        prediction,
        target
    ):
        return torch.mean(
            torch.abs(prediction - target)
        )

    @staticmethod
    def rmse(
        prediction,
        target
    ):
        return torch.sqrt(
            torch.mean(
                (prediction - target) ** 2
            )
        )

    @staticmethod
    def relative_error(
        prediction,
        target,
        eps=1e-8
    ):
        return torch.mean(
            torch.abs(prediction - target)
            /
            (torch.abs(target) + eps)
        )

    @staticmethod
    def log_mae(
        prediction,
        target
    ):
        return torch.mean(
            torch.abs(
                torch.log(prediction) - torch.log(target)
            )
        )


class TrajectoryMetrics:

    @staticmethod
    def mae(
        prediction,
        target
    ):
        return torch.mean(
            torch.abs(prediction - target)
        )

    @staticmethod
    def rmse(
        prediction,
        target
    ):
        return torch.sqrt(
            torch.mean(
                (prediction - target) ** 2
            )
        )

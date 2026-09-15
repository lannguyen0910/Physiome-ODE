import torch


class TrajectoryMetrics:
    def __init__(
        self,
        eps=1e-8,
        methods=None,
    ):

        self.eps = float(eps)

        if methods is None:

            methods = [
                "mae",
                "rmse",
                "normalized_rmse",
            ]

        self.methods = list(
            methods
        )

    @staticmethod
    def mae(
        prediction,
        target,
    ):

        return torch.mean(
            torch.abs(
                prediction - target
            )
        )

    @staticmethod
    def rmse(
        prediction,
        target,
    ):

        return torch.sqrt(
            torch.mean(
                (
                    prediction - target
                ) ** 2
            )
        )

    def normalized_rmse(
        self,
        prediction,
        target,
    ):

        rmse = self.rmse(
            prediction,
            target,
        )

        scale = (
            target.max()
            -
            target.min()
        )

        return (
            rmse
            /
            (
                scale
                + self.eps
            )
        )

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

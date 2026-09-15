import torch


class ParameterMetrics:
    def __init__(
        self,
        tolerance=0.1,
        eps=1e-8,
        parameter_scale=None,
        methods=None,
    ):

        self.tolerance = float(
            tolerance
        )

        self.eps = float(eps)

        self.parameter_scale = (
            parameter_scale
        )

        # Default: metrics that do not require
        # an externally supplied parameter scale.
        if methods is None:

            methods = [
                "mae",
                "rmse",
                "relative_error",
                "per_parameter_relative_error",
                "per_parameter_accuracy",
                "overall_parameter_accuracy",
                "log_mae",
            ]

        self.methods = list(
            methods
        )

    # =====================================================
    # BASIC METRICS
    # =====================================================

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

    # =====================================================
    # RELATIVE ERROR
    # =====================================================

    def relative_error(
        self,
        prediction,
        target,
    ):

        relative_error = (
            torch.abs(
                prediction - target
            )
            /
            (
                torch.abs(target)
                + self.eps
            )
        )

        return torch.mean(
            relative_error
        )

    # =====================================================
    # NORMALIZED ERROR
    # =====================================================

    def normalized_error(
        self,
        prediction,
        target,
    ):

        if self.parameter_scale is None:

            raise ValueError(
                "parameter_scale is required "
                "for normalized_error."
            )

        scale = torch.as_tensor(
            self.parameter_scale,
            dtype=prediction.dtype,
            device=prediction.device,
        )

        error = torch.abs(
            prediction - target
        )

        normalized_error = (
            error
            /
            (
                scale + self.eps
            )
        )

        return torch.mean(
            normalized_error
        )

    # =====================================================
    # PER PARAMETER
    # =====================================================

    def per_parameter_relative_error(
        self,
        prediction,
        target,
    ):

        relative_error = (
            torch.abs(
                prediction - target
            )
            /
            (
                torch.abs(target)
                + self.eps
            )
        )

        return torch.mean(
            relative_error,
            dim=0,
        )

    def per_parameter_accuracy(
        self,
        prediction,
        target,
    ):

        relative_error = (
            torch.abs(
                prediction - target
            )
            /
            (
                torch.abs(target)
                + self.eps
            )
        )

        correct = (
            relative_error
            <= self.tolerance
        ).float()

        return torch.mean(
            correct,
            dim=0,
        )

    def overall_parameter_accuracy(
        self,
        prediction,
        target,
    ):

        accuracy = (
            self.per_parameter_accuracy(
                prediction,
                target,
            )
        )

        return torch.mean(
            accuracy
        )

    # =====================================================
    # LOG MAE
    # =====================================================

    def log_mae(
        self,
        prediction,
        target,
    ):

        prediction = torch.clamp(
            prediction,
            min=self.eps,
        )

        target = torch.clamp(
            target,
            min=self.eps,
        )

        return torch.mean(
            torch.abs(
                torch.log(prediction)
                -
                torch.log(target)
            )
        )

    # =====================================================
    # COMPUTE SELECTED METRICS
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

            value = method(
                prediction,
                target,
            )

            results[method_name] = value

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

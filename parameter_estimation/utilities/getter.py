from typing import Any, Dict, Optional


def build_from_registry(
    config: Any,
    registry,
    **extra_kwargs,
):
    """
    Build an object from:

        config:
            name: ...
            params:
                ...

    Example:

        encoder:
            name: gru
            params:
                hidden_dim: 128
    """

    if config is None:
        raise ValueError(
            "Configuration cannot be None."
        )

    name = config["name"]

    params = dict(
        config.get(
            "params",
            {},
        )
    )

    params.update(
        extra_kwargs
    )

    cls = registry.get(name)

    return cls(**params)

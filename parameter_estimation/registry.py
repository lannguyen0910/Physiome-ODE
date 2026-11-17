from typing import Any, Dict, Iterable, Iterator, Tuple

from tabulate import tabulate


class Registry(Iterable[Tuple[str, Any]]):
    """
    Generic name -> object registry.

    Example:

        BACKBONE_REGISTRY = Registry("BACKBONE")

        @BACKBONE_REGISTRY.register()
        class MyBackbone:
            ...

    Or:

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name: str) -> None:
        self._name: str = name
        self._obj_map: Dict[str, Any] = {}

    def _do_register(
        self,
        name: str,
        obj: Any,
        override: bool = False,
    ) -> None:

        if not override:
            assert name not in self._obj_map, (
                f"An object named '{name}' was already registered "
                f"in '{self._name}' registry!"
            )

        self._obj_map[name] = obj

    def register(
        self,
        obj: Any = None,
        prefix: str = "",
        override: bool = False,
    ) -> Any:
        """
        Register an object.

        Can be used as:

            @REGISTRY.register()
            class MyClass:
                ...

        or:

            REGISTRY.register(MyClass)
        """

        if obj is None:

            def deco(func_or_class: Any) -> Any:
                name = func_or_class.__name__
                self._do_register(
                    prefix + name,
                    func_or_class,
                    override,
                )
                return func_or_class

            return deco

        name = obj.__name__

        self._do_register(
            prefix + name,
            obj,
            override,
        )

        return obj

    def get(self, name: str) -> Any:

        ret = self._obj_map.get(name)

        if ret is None:
            raise KeyError(
                f"No object named '{name}' found in "
                f"'{self._name}' registry!"
            )

        return ret

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        return iter(self._obj_map.items())

    def __repr__(self) -> str:

        table = tabulate(
            self._obj_map.items(),
            headers=["Names", "Objects"],
            tablefmt="fancy_grid",
        )

        return f"Registry of {self._name}:\n{table}"

    __str__ = __repr__

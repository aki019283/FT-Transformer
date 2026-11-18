"""各插件类别共享的类型化注册表实现。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar


T = TypeVar("T")


class TypedRegistry(Generic[T]):
    """维护单一插件类别的名称到 Factory 映射。"""

    def __init__(self, category: str) -> None:
        """创建一个空的类别注册表。"""

        self.category = category
        self._factories: dict[str, Callable[..., T]] = {}

    def Register(self, name: str, factory: Callable[..., T]) -> None:
        """显式注册一个 Factory。"""

        if not name:
            raise ValueError("Registry name cannot be empty")
        if name in self._factories:
            raise ValueError(f"Duplicate {self.category} registration: {name}")
        self._factories[name] = factory

    def Create(self, name: str, *args, **kwargs) -> T:
        """按名称调用 Factory。"""

        try:
            factory = self._factories[name]
        except KeyError as error:
            available = ", ".join(sorted(self._factories)) or "none"
            raise ValueError(
                f"Unknown {self.category}: {name}. Available: {available}"
            ) from error
        return factory(*args, **kwargs)

    def Names(self) -> tuple[str, ...]:
        """返回稳定排序的已注册名称。"""

        return tuple(sorted(self._factories))


__all__ = ["TypedRegistry"]

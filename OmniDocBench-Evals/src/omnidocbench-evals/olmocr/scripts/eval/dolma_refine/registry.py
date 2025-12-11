import re
from typing import (
    Callable,
    Dict,
    Generator,
    Generic,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    overload,
)

T = TypeVar("T")
R = TypeVar("R")


class BaseRegistry(Generic[T]):
    """A registry for objects."""

    _registry_of_registries: Dict[str, Type["BaseRegistry"]] = {}
    _registry_storage: Dict[str, Tuple[T, Optional[str]]]

    @classmethod
    def _add_to_registry_of_registries(cls) -> None:
        name = cls.__name__
        if name not in cls._registry_of_registries:
            cls._registry_of_registries[name] = cls

    @classmethod
    def registries(cls) -> Generator[Tuple[str, Type["BaseRegistry"]], None, None]:
        """Yield all registries in the registry of registries."""
        yield from sorted(cls._registry_of_registries.items())

    @classmethod
    def _get_storage(cls) -> Dict[str, Tuple[T, Optional[str]]]:
        if not hasattr(cls, "_registry_storage"):
            cls._registry_storage = {}
        return cls._registry_storage  # pyright: ignore

    @classmethod
    def items(cls) -> Generator[Tuple[str, T], None, None]:
        """Yield all items in the registry."""
        yield from sorted((n, t) for (n, (t, _)) in cls._get_storage().items())

    @classmethod
    def items_with_description(cls) -> Generator[Tuple[str, T, Optional[str]], None, None]:
        """Yield all items in the registry with their descriptions."""
        yield from sorted((n, t, d) for (n, (t, d)) in cls._get_storage().items())

    @classmethod
    def add(cls, name: str, desc: Optional[str] = None) -> Callable[[R], R]:
        """Add a class to the registry."""

        # Add the registry to the registry of registries
        cls._add_to_registry_of_registries()

        def _add(
            inner_self: T,
            inner_name: str = name,
            inner_desc: Optional[str] = desc,
            inner_cls: Type[BaseRegistry] = cls,
        ) -> T:
            """Add a tagger to the registry using tagger_name as the name."""

            existing = inner_cls.get(inner_name, raise_on_missing=False)

            if existing and existing != inner_self:
                if inner_self.__module__ == "__main__":
                    return inner_self

                raise ValueError(f"Tagger {inner_name} already exists")
            inner_cls._get_storage()[inner_name] = (inner_self, inner_desc)
            return inner_self

        return _add  # type: ignore

    @classmethod
    def remove(cls, name: str) -> bool:
        """Remove a tagger from the registry."""
        if name in cls._get_storage():
            cls._get_storage().pop(name)
            return True
        return False

    @classmethod
    def has(cls, name: str) -> bool:
        """Check if a tagger exists in the registry."""
        return name in cls._get_storage()

    @overload
    @classmethod
    def get(cls, name: str) -> T: ...

    @overload
    @classmethod
    def get(cls, name: str, raise_on_missing: Literal[True]) -> T: ...

    @overload
    @classmethod
    def get(cls, name: str, raise_on_missing: Literal[False]) -> Optional[T]: ...

    @classmethod
    def get(cls, name: str, raise_on_missing: bool = True) -> Optional[T]:
        """Get a tagger from the registry; raise ValueError if it doesn't exist."""

        matches = [registered for registered in cls._get_storage() if re.match(registered, name)]

        if len(matches) > 1:
            raise ValueError(f"Multiple taggers match {name}: {', '.join(matches)}")

        elif len(matches) == 0:
            if raise_on_missing:
                tagger_names = ", ".join([tn for tn, _ in cls.items()])
                raise ValueError(f"Unknown tagger {name}; available taggers: {tagger_names}")
            return None

        else:
            name = matches[0]
            t, _ = cls._get_storage()[name]
            return t

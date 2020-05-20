from typing import Optional
from imc.types import GenericType


class AttributeNotSetError(Exception):
    pass


def cast(arg: Optional[GenericType]) -> GenericType:
    """Remove `Optional` from `T`."""
    if arg is None:
        raise AttributeNotSetError("Attribute cannot be None!")
    return arg

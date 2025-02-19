# --------------------------------------
import typing as tp

# --------------------------------------
import numpy as np

# --------------------------------------
from functools import partial

# --------------------------------------
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.parameterTypes import GroupParameter

# --------------------------------------
from pyrception.visual.utils.types import AuxEnum
from pyrception.gui.param.enumparam import EnumParameter
from pyrception.gui.param.rf_param import RFParameterGroup
from pyrception.gui.param.syncparam import SyncParameter


def title_name(fun: tp.Callable):

    def _inner(title: str = None, *args, **kwargs):

        if title is None:
            title = kwargs.pop("title", None)
            if title is None and len(args) > 0:
                title, args = args[0], args[1:]
            else:
                title = ""

        kwargs["title"] = title
        if "name" not in kwargs:
            kwargs["name"] = title.lower().replace(" ", "_").replace(".", "")

        return fun(*args, **kwargs)

    return _inner


@title_name
def make_group(
    children: tp.List = None,
    **kwargs,
) -> Parameter:
    """
    Group parameter.

    Args:

        children (tp.List, optional):
            Sub-parameters to be grouped. Defaults to None.

    Returns:
        Parameter:
            Group parameter.
    """
    if children is None:
        children = []

    p = {
        "expanded": False,
        "type": "group",
        "children": children,
    }
    p.update(kwargs)
    return Parameter.create(**p)


@title_name
def make_int(value: int, **kwargs) -> Parameter:
    p = {
        "type": "int",
        "compactHeight": False,
        "value": value,
    }
    p.update(kwargs)
    return Parameter.create(**p)


@title_name
def make_float(value: float, **kwargs) -> Parameter:
    p = {
        "type": "float",
        "value": value,
        "compactHeight": False,
        "limits": [0.1, 10],
        "step": 0.1,
    }
    p.update(kwargs)
    return Parameter.create(**p)


@title_name
def make_bool(value: bool = True, **kwargs) -> Parameter:
    p = {
        "type": "bool",
        "value": value,
    }
    p.update(kwargs)
    return Parameter.create(**p)


@title_name
def make_str(**kwargs) -> Parameter:
    p = {
        "type": "str",
    }
    p.update(kwargs)
    return Parameter.create(**p)


@title_name
def make_list(
    limits: tp.List,
    **kwargs,
) -> Parameter:
    p = {
        "type": "list",
        "limits": limits,
    }
    p.update(kwargs)
    return Parameter.create(**p)


@title_name
def make_enum(
    evalue: AuxEnum,
    **kwargs,
) -> EnumParameter:
    p = {
        "type": "enum",
        "limits": evalue.names(),
        "value": evalue,
    }
    p.update(kwargs)
    return Parameter.create(**p)


def make_rf_params(**kwargs) -> RFParameterGroup:

    kwargs.setdefault("type", "rf_params")
    kwargs.setdefault("name", "rf_params")
    kwargs.setdefault("title", "RF parameters")

    return Parameter.create(**kwargs)


@title_name
def make_sync_params(
    cat: tp.Callable,
    value: tp.Iterable,
    limits: tp.Iterable,
    default: bool = True,
    names: tp.List[str] = None,
    **kwargs,
) -> SyncParameter:

    if names is None:
        names = ["x", "y"]

    p = {
        "type": "syncparam",
        "name": "Sync",
        "cat": cat,
        "value": value,
        "limits": limits,
        "default": default,
        "names": names,
    }
    p.update(kwargs)
    return Parameter.create(**p)


def to_dict(
    param: Parameter,
    titles: bool = False,
):

    children = {}

    for name, obj in param.names.items():
        key = obj.title() if titles else name
        if isinstance(obj, SyncParameter):
            children[key] = obj.to_dict()

        elif isinstance(obj, GroupParameter):

            # Recurse over the object's children
            children[key] = to_dict(obj, titles)

        else:
            # Just dump the value
            children[key] = obj.value()

    return children

"""wandb compatibility helpers.

wandb 0.17+ scans the value passed to ``wandb.init(config=...)`` and calls
``v.get("_type")`` on every value it considers dict-like. pyhocon's
``ConfigTree.get(key)`` is not dict-compatible — without an explicit default
it raises ``ConfigMissingException`` instead of returning ``None``. Passing a
``ConfigTree`` (or any object containing nested ``ConfigTree`` values) into
``wandb.init`` therefore crashes during init, regardless of ``mode=``.

Convert config trees to plain ``dict`` / ``list`` before handing them to
wandb.
"""
from collections.abc import Mapping


def to_plain_dict(value):
    if isinstance(value, Mapping):
        return {k: to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_dict(v) for v in value]
    return value

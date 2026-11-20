"""Serialization-safe successor of circuit_framework_v1.

All numerical propagation, fixed words, thresholds, seeds, and correction
rules are imported unchanged from v1.  This successor only recursively encodes
complex values for durable JSON records and emits a per-module trial sink.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

import circuit_framework_v1 as _v1
from circuit_framework_v1 import *  # noqa: F401,F403 - frozen numerical implementation
from circuit_framework_v1 import m8 as m8


_durable_trial_sink: Callable[[dict[str, Any]], None] | None = None


def jsonable(value: Any) -> Any:
    """Recursively turn arrays and complex values into JSON-safe objects."""
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def set_durable_trial_sink(sink: Callable[[dict[str, Any]], None] | None) -> None:
    """Install a runner-owned exclusive-write sink for module trial records."""
    global _durable_trial_sink
    _durable_trial_sink = sink


def construct_target_time_ablation(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run the frozen v1 correction while persisting start/finish trial events."""
    label = kwargs.get("label", args[0] if args else None)
    if _durable_trial_sink is not None:
        _durable_trial_sink({"event": "baseline_module_started", "label": label})
    try:
        result = _v1.construct_target_time_ablation(*args, **kwargs)
    except Exception as exc:
        if _durable_trial_sink is not None:
            _durable_trial_sink({
                "event": "baseline_module_exception",
                "label": label,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            })
        raise
    if _durable_trial_sink is not None:
        _durable_trial_sink({
            "event": "baseline_module_completed",
            "label": label,
            "record": result,
        })
    return result


__all__ = list(_v1.__all__) + ["set_durable_trial_sink"]

# ctx is an immutable object (a frozen dataclass or dict). It is supplied at
# run time and shared across stages; a Stage may carry its own ctx to override it.

from __future__ import annotations

from collections.abc import Callable
from graphlib import TopologicalSorter
from typing import Any, TypeAlias, overload

Ctx: TypeAlias = Any
StageFn: TypeAlias = Callable[[Ctx], Any]
Operand: TypeAlias = "Stage | list[Stage] | DAG"


class Stage:
    """Wraps `fn(ctx)`. If `fn` is not callable it is a static stage whose
    result is `fn` itself, ignoring ctx."""

    def __init__(self, name: str, fn: StageFn | Any, ctx: Ctx | None = None) -> None:
        self.name = name
        self.fn = fn
        self.ctx = ctx

    def run(self, ctx: Ctx | None = None) -> Any:
        if not callable(self.fn):  # static stage: result is the value itself
            return self.fn
        return self.fn(self.ctx if self.ctx is not None else ctx)

    def __rshift__(self, other: Operand) -> DAG:
        return _compose(self, other)

    def __rrshift__(self, other: Operand) -> DAG:
        return _compose(other, self)

    def __repr__(self) -> str:
        return f"Stage({self.name!r})"


@overload
def stage(fn: StageFn | Any, *, name: str | None = ..., ctx: Ctx | None = ...) -> Stage: ...
@overload
def stage(fn: None = ..., *, name: str | None = ..., ctx: Ctx | None = ...) -> Callable[[StageFn], Stage]: ...
def stage(
    fn: StageFn | Any | None = None,
    *,
    name: str | None = None,
    ctx: Ctx | None = None,
) -> Stage | Callable[[StageFn], Stage]:
    """Wrap a function `fn(ctx)` into a Stage, like Dask's `delayed`. The stage
    name defaults to the function's `__name__`. A non-callable `fn` becomes a
    static stage and then needs an explicit `name`.

    Usable bare or parameterized::

        @stage
        def load(ctx): ...

        @stage(name="train", ctx=cfg)
        def fit(ctx): ...

        s = stage(load, ctx=cfg)
        const = stage(42, name="seed")
    """

    def make(f: StageFn | Any) -> Stage:
        n = name or getattr(f, "__name__", None)
        if n is None:
            raise ValueError("static stage requires an explicit name")
        return Stage(n, f, ctx)

    return make(fn) if fn is not None else make


class DAG:
    """A composed graph of stages, holding only the dependency map. The build
    cursor (frontier) is derived from the graph as its sink stages, so a DAG is
    an immutable value: composition never mutates it, and it can be reused or
    composed further freely."""

    def __init__(self, deps: dict[Stage, set[Stage]] | None = None) -> None:
        self.deps: dict[Stage, set[Stage]] = dict(deps) if deps else {}

    def run(self, ctx: Ctx | None = None) -> dict[str, Any]:
        return {s.name: s.run(ctx) for s in self._toposort()}

    def _toposort(self) -> list[Stage]:
        # deps maps each stage to its predecessors — exactly TopologicalSorter's
        # input shape. CycleError subclasses ValueError with a "cycle" message.
        return list(TopologicalSorter(self.deps).static_order())

    def __rshift__(self, other: Operand) -> DAG:
        return _compose(self, other)

    def __rrshift__(self, other: Operand) -> DAG:
        return _compose(other, self)

    def __repr__(self) -> str:
        return f"DAG(stages={[s.name for s in self.deps]})"


def _normalize(operand: Operand) -> dict[Stage, set[Stage]]:
    """Return a fresh deps map for any composable operand: a Stage, a list of
    Stages, or a DAG."""
    if isinstance(operand, Stage):
        return {operand: set()}
    if isinstance(operand, list):
        if not all(isinstance(s, Stage) for s in operand):
            raise TypeError(f"cannot compose with {operand!r}")
        return {s: set() for s in operand}
    if isinstance(operand, DAG):
        return {s: set(preds) for s, preds in operand.deps.items()}
    raise TypeError(f"cannot compose with {operand!r}")


def _sinks(deps: dict[Stage, set[Stage]]) -> list[Stage]:
    has_succ = set().union(*deps.values()) if deps else set()
    return [s for s in deps if s not in has_succ]


def _compose(left: Operand, right: Operand) -> DAG:
    deps = _normalize(left)
    frontier = _sinks(deps)
    for s, preds in _normalize(right).items():
        merged = deps.setdefault(s, set())
        merged.update(preds)
        if not preds:  # source stage of `right` depends on left's frontier
            merged.update(frontier)
    return DAG(deps)

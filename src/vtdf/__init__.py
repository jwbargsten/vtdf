# ctx is an immutable object (a frozen dataclass or dict). It is supplied at
# run time and shared across stages; a Stage may carry its own ctx to override it.
#
# Call contract (unwritten — fn must have the matching arity): a stage's fn is
# called with its predecessors' results in dependency order, then ctx last:
#   a >> b       -> b is fn(result_a, ctx)
#   [a, b] >> c  -> c is fn(result_a, result_b, ctx)
#   source stage -> fn(ctx)

from __future__ import annotations

from collections.abc import Callable
from graphlib import TopologicalSorter
from typing import Any, TypeAlias, overload

Ctx: TypeAlias = Any
StageFn: TypeAlias = Callable[..., Any]
Operand: TypeAlias = "Stage | list[Stage] | DAG"


class Stage:
    """Wraps a function called as `fn(*predecessor_results, ctx)`; a source stage
    is `fn(ctx)`. If `fn` is not callable it is a static stage whose result is
    `fn` itself, ignoring its inputs."""

    def __init__(self, name: str, fn: Any, ctx: Ctx | None = None) -> None:
        self.name = name
        self.fn = fn
        self.ctx = ctx

    def run(self, *results: Any, ctx: Ctx | None = None) -> Any:
        if not callable(self.fn):  # static stage: result is the value itself
            return self.fn
        return self.fn(*results, self.ctx if self.ctx is not None else ctx)

    def __rshift__(self, other: Operand) -> DAG:
        return _compose(self, other)

    def __rrshift__(self, other: Operand) -> DAG:
        return _compose(other, self)

    def __repr__(self) -> str:
        return f"Stage({self.name!r})"


@overload
def stage(fn: StageFn, *, name: str | None = ..., ctx: Ctx | None = ...) -> Stage: ...
@overload
def stage(fn: None = ..., *, name: str | None = ..., ctx: Ctx | None = ...) -> Callable[[StageFn], Stage]: ...
def stage(
    fn: Any = None,
    *,
    name: str | None = None,
    ctx: Ctx | None = None,
) -> Stage | Callable[[StageFn], Stage]:
    """Wrap a function into a Stage, like Dask's `delayed`. The stage name
    defaults to the function's `__name__`. See the module docstring for the
    `fn(*predecessor_results, ctx)` call contract. A non-callable `fn` becomes a
    static stage and then needs an explicit `name`.

    Usable bare or parameterized::

        @stage
        def load(ctx): ...

        @stage(name="train", ctx=cfg)
        def fit(ctx): ...

        s = stage(load, ctx=cfg)
        const = stage(42, name="seed")
    """

    def make(f: Any) -> Stage:
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

    def __init__(self, deps: dict[Stage, list[Stage]] | None = None) -> None:
        self.deps: dict[Stage, list[Stage]] = (
            {s: list(preds) for s, preds in deps.items()} if deps else {}
        )

    def run(self, ctx: Ctx | None = None) -> Any:
        """Execute in dependency order, threading each stage's result to its
        successors. Returns the last job's result: the sole sink's value, or a
        tuple of all sink results (in sink order) when there is more than one."""
        results: dict[Stage, Any] = {}
        for s in self._toposort():
            results[s] = s.run(*(results[p] for p in self.deps[s]), ctx=ctx)
        sinks = _sinks(self.deps)
        if not sinks:
            return None
        return results[sinks[0]] if len(sinks) == 1 else tuple(results[s] for s in sinks)

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


def _normalize(operand: Operand) -> dict[Stage, list[Stage]]:
    """Return a fresh, order-preserving deps map for any composable operand: a
    Stage, a list of Stages, or a DAG."""
    if isinstance(operand, Stage):
        return {operand: []}
    if isinstance(operand, list):
        if not all(isinstance(s, Stage) for s in operand):
            raise TypeError(f"cannot compose with {operand!r}")
        return {s: [] for s in operand}
    if isinstance(operand, DAG):
        return {s: list(preds) for s, preds in operand.deps.items()}
    raise TypeError(f"cannot compose with {operand!r}")


def _sinks(deps: dict[Stage, list[Stage]]) -> list[Stage]:
    has_succ = set().union(*deps.values()) if deps else set()
    return [s for s in deps if s not in has_succ]


def _compose(left: Operand, right: Operand) -> DAG:
    deps = _normalize(left)
    frontier = _sinks(deps)
    for s, preds in _normalize(right).items():
        merged = deps.setdefault(s, [])
        for p in (preds or frontier):  # empty preds => source stage => left's frontier
            if p not in merged:
                merged.append(p)
    return DAG(deps)

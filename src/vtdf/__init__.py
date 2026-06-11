# ctx is an immutable object (a frozen dataclass or dict). It is supplied at
# run time and shared across stages; a Stage may carry its own ctx to override it.
#
# Call contract (unwritten — fn must have the matching arity): a stage's fn is
# called with its predecessors' results in dependency order, then ctx last:
#   a >> b       -> b is fn(result_a, ctx)
#   [a, b] >> c  -> c is fn(result_a, result_b, ctx)
#   source stage -> fn(ctx)

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from typing import Any, TypeAlias, overload

Ctx: TypeAlias = Any
StageFn: TypeAlias = Callable[..., Any]
Node: TypeAlias = "Stage | Artifact"
Operand: TypeAlias = "Node | Sequence[Node] | DAG"


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


@dataclass(eq=False)
class Artifact:
    """A remote resource (a GCS object, a BigQuery table, ...) that sits in a DAG
    as a source or sink. As a node it yields itself: a consuming stage receives
    the Artifact (read its `uri`); a producing stage's result is ignored, the
    sink being the Artifact itself. Hashed by identity, like Stage."""

    name: str
    uri: str
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def run(self, *results: Any, ctx: Ctx | None = None) -> Artifact:
        return self

    def __rshift__(self, other: Operand) -> DAG:
        return _compose(self, other)

    def __rrshift__(self, other: Operand) -> DAG:
        return _compose(other, self)

    def __repr__(self) -> str:
        return f"Artifact({self.name!r})"


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

    def __init__(self, deps: dict[Node, list[Node]] | None = None) -> None:
        self.deps: dict[Node, list[Node]] = (
            {s: list(preds) for s, preds in deps.items()} if deps else {}
        )
        seen: set[str] = set()
        for s in self.deps:
            if s.name in seen:
                raise ValueError(f"duplicate node name: {s.name!r}")
            seen.add(s.name)

    def run(self, ctx: Ctx | None = None) -> Any:
        """Execute the whole DAG in dependency order, threading each stage's
        result to its successors. Returns the last job's result: the sole sink's
        value, or a tuple of all sink results (in sink order) when there is more
        than one. Use `run_collect` for every stage's output."""
        results = self._execute(ctx)
        sinks = _sinks(self.deps)
        if not sinks:
            return None
        return results[sinks[0]] if len(sinks) == 1 else tuple(results[s] for s in sinks)

    def run_collect(self, ctx: Ctx | None = None) -> dict[str, Any]:
        """Execute the whole DAG and return `{stage_name: result}` for all stages."""
        results = self._execute(ctx)
        return {s.name: r for s, r in results.items()}

    def _execute(self, ctx: Ctx | None) -> dict[Node, Any]:
        results: dict[Node, Any] = {}
        for s in self._toposort():
            results[s] = s.run(*(results[p] for p in self.deps[s]), ctx=ctx)
        return results

    def _toposort(self) -> list[Node]:
        # deps maps each node to its predecessors — exactly TopologicalSorter's
        # input shape. CycleError subclasses ValueError with a "cycle" message.
        return list(TopologicalSorter(self.deps).static_order())

    def __rshift__(self, other: Operand) -> DAG:
        return _compose(self, other)

    def __rrshift__(self, other: Operand) -> DAG:
        return _compose(other, self)

    def __repr__(self) -> str:
        return f"DAG(stages={[s.name for s in self.deps]})"


def _normalize(operand: Operand) -> dict[Node, list[Node]]:
    """Return a fresh, order-preserving deps map for any composable operand: a
    node (Stage or Artifact), a list of nodes, or a DAG."""
    if isinstance(operand, (Stage, Artifact)):
        return {operand: []}
    if isinstance(operand, list):
        if not all(isinstance(s, (Stage, Artifact)) for s in operand):
            raise TypeError(f"cannot compose with {operand!r}")
        return {s: [] for s in operand}
    if isinstance(operand, DAG):
        return {s: list(preds) for s, preds in operand.deps.items()}
    raise TypeError(f"cannot compose with {operand!r}")


def _sinks(deps: dict[Node, list[Node]]) -> list[Node]:
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

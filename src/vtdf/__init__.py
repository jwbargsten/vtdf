# ctx is an immutable object (a frozen dataclass or dict). It is supplied at
# run time and shared across stages; a Stage may carry its own ctx to override it.

from collections import deque


class Stage:
    def __init__(self, name, fn, ctx=None):
        self.name = name
        self.fn = fn
        self.ctx = ctx

    def run(self, ctx=None):
        return self.fn(self.ctx if self.ctx is not None else ctx)

    def __rshift__(self, other):
        return _compose(self, other)

    def __rrshift__(self, other):
        return _compose(other, self)

    def __repr__(self):
        return f"Stage({self.name!r})"


class DAG:
    """A composed graph of stages, holding only the dependency map. The build
    cursor (frontier) is derived from the graph as its sink stages, so a DAG is
    an immutable value: composition never mutates it, and it can be reused or
    composed further freely."""

    def __init__(self, deps=None):
        self.deps = dict(deps) if deps else {}

    def run(self, ctx=None):
        return {s.name: s.run(ctx) for s in self._toposort()}

    def _toposort(self):
        indeg = {s: len(preds) for s, preds in self.deps.items()}
        succ = {s: [] for s in self.deps}
        for s, preds in self.deps.items():
            for p in preds:
                succ[p].append(s)
        queue = deque(s for s, d in indeg.items() if d == 0)
        order = []
        while queue:
            s = queue.popleft()
            order.append(s)
            for t in succ[s]:
                indeg[t] -= 1
                if indeg[t] == 0:
                    queue.append(t)
        if len(order) != len(self.deps):
            raise ValueError("cycle detected in DAG")
        return order

    def __rshift__(self, other):
        return _compose(self, other)

    def __rrshift__(self, other):
        return _compose(other, self)

    def __repr__(self):
        return f"DAG(stages={[s.name for s in self.deps]})"


def _normalize(operand):
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


def _sinks(deps):
    has_succ = set().union(*deps.values()) if deps else set()
    return [s for s in deps if s not in has_succ]


def _compose(left, right):
    deps = _normalize(left)
    frontier = _sinks(deps)
    for s, preds in _normalize(right).items():
        merged = deps.setdefault(s, set())
        merged.update(preds)
        if not preds:  # source stage of `right` depends on left's frontier
            merged.update(frontier)
    return DAG(deps)

# ctx is an immutable object (a frozen dataclass or dict) shared across stages.

from collections import deque


class Stage:
    def __init__(self, name, fn, ctx):
        self.name = name
        self.fn = fn
        self.ctx = ctx

    def __rshift__(self, other):
        return _compose(self, other)

    def __rrshift__(self, other):
        return _compose(other, self)


class DAG:
    def __init__(self):
        self.deps = {}        # stage -> set of predecessor stages
        self.frontier = []    # most recently added stages; next `>>` depends on these

    def __rshift__(self, other):
        return _compose(self, other)

    def __rrshift__(self, other):
        return _compose(other, self)

    def run(self):
        for stage in self._toposort():
            stage.fn(stage.ctx)

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


def _stages(operand):
    if isinstance(operand, Stage):
        return [operand]
    if isinstance(operand, list):
        return list(operand)
    raise TypeError(f"cannot compose with {operand!r}")


def _to_dag(operand):
    if isinstance(operand, DAG):
        return operand
    dag = DAG()
    dag.frontier = _stages(operand)
    for s in dag.frontier:
        dag.deps[s] = set()
    return dag


def _compose(left, right):
    dag = _to_dag(left)
    rstages = _stages(right)
    for s in rstages:
        dag.deps.setdefault(s, set()).update(dag.frontier)
    dag.frontier = rstages
    return dag

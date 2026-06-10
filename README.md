# Very Tiny DAG Framework

`vtdf` composes computation stages into a directed acyclic graph using
Airflow-style `>>` syntax, then runs them in dependency order.

## Overview

A `Stage(name, fn, ctx)` wraps a function `fn(ctx)`. `ctx` is an immutable
object (a frozen dataclass or dict) shared across stages; a stage may carry its
own `ctx` to override the one passed to `run()`.

Stages compose with `>>`:

- `a >> b` — `b` depends on `a`.
- `[a, b] >> c` — fan-in: `c` depends on both `a` and `b`.
- `c >> [d, e]` — fan-out: `d` and `e` depend on `c`.

A list on either side means those stages run in parallel (no ordering between
them). Composition returns a `DAG`, which is itself composable, so sub-graphs
combine freely:

```python
from dataclasses import dataclass
from vtdf import Stage

@dataclass(frozen=True)
class Ctx:
    learning_rate: float

ctx = Ctx(learning_rate=0.3)
a, b, c, d, e = (Stage(n, lambda c: n, ctx) for n in "abcde")

dag = [a, b] >> c >> [d, e]
results = dag.run()   # {"a": ..., "b": ..., "c": ..., "d": ..., "e": ...}
```

`run()` returns a `dict` mapping each stage name to its return value.

## How it works

### Data model

A `DAG` holds a single field, `deps: dict[Stage, set[Stage]]`, mapping each
stage to its set of predecessors (the stages that must run before it). That is
the entire graph — there is no separate node list or mutable build cursor.

The **frontier** (where the next `>>` attaches) is *derived* from the graph as
its sink stages: stages that appear as nobody's predecessor. Because the
frontier is computed, never stored, a `DAG` is an immutable value — composition
builds a new map and never mutates its operands, so a DAG can be reused or
composed further freely.

### Composition (`>>`)

Each operand — a `Stage`, a `list[Stage]`, or a `DAG` — is first `_normalize`d
into a deps map:

- `Stage` → `{stage: ∅}`
- `list[Stage]` → `{s: ∅ for s in list}` (parallel, no inter-dependencies)
- `DAG` → a fresh copy of its deps map

`_compose(left, right)` then:

1. Normalizes `left` into a new map `deps`.
2. Computes `frontier = sinks(deps)` — the stages with no successors in `left`.
3. Merges every stage of `right` into `deps`. Each **source** stage of `right`
   (one with no predecessors of its own) gains the whole `frontier` as its
   predecessors; that is the edge that wires the two operands together.

This makes the operations associative over sub-DAGs: `(a >> b) >> (c >> d)`
attaches `c`'s sources to `b` (the sink of the left DAG), yielding the linear
chain `a → b → c → d`.

### Execution

`run()` linearizes the graph with **Kahn's algorithm** (`_toposort`):

1. Compute the in-degree of each stage (its number of predecessors).
2. Seed a queue with all zero-in-degree stages (the sources).
3. Repeatedly pop a stage, append it to the order, and decrement the in-degree
   of each successor; when a successor reaches zero, enqueue it.
4. If the produced order is shorter than the stage count, a cycle exists →
   raise `ValueError("cycle detected in DAG")`.

Stages then run in that topological order, each invoked with its own `ctx` if it
carries one, otherwise the `ctx` passed to `run()`. Results are collected by
stage name.

Execution is currently sequential; the topological order respects all
dependencies, and parallel branches are simply emitted in an unspecified order.

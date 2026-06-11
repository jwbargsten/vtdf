# Very Tiny DAG Framework

`vtdf` composes computation stages into a directed acyclic graph using
Airflow-style `>>` syntax, then runs them in dependency order.

## Overview

A `Stage(name, fn, ctx)` wraps a function. A source stage is called `fn(ctx)`; a
stage with predecessors is called `fn(*predecessor_results, ctx)` — its inputs in
dependency order, then `ctx` last. `ctx` is an immutable object (a frozen
dataclass or dict) shared across stages; a stage may carry its own `ctx` to
override the one passed to `run()`.

Stages compose with `>>`:

- `a >> b` — `b` depends on `a`.
- `[a, b] >> c` — fan-in: `c` depends on both `a` and `b`.
- `c >> [d, e]` — fan-out: `d` and `e` depend on `c`.

A list on either side means those stages run in parallel (no ordering between
them). Composition returns a `DAG`, which is itself composable, so sub-graphs
combine freely:

```python
from vtdf import Stage

def source(ctx):       return 1
def combine(x, y, ctx): return x + y   # fan-in: receives a and b results
def inc(prev, ctx):     return prev + 1

a, b = Stage("a", source), Stage("b", source)
c = Stage("c", combine)
d, e = Stage("d", inc), Stage("e", inc)

dag = [a, b] >> c >> [d, e]

dag.run()          # -> (3, 3): the sink results (d, e), in sink order
dag.run_collect()  # -> {"a": 1, "b": 1, "c": 2, "d": 3, "e": 3}
```

Two run methods, both taking an optional `ctx`:

- `run()` returns the **last job's** result — the sole sink's value, or a tuple of
  all sink values (in sink order) when there is more than one.
- `run_collect()` returns a `dict` mapping **every** stage name to its result.

Stage names must be unique within a DAG; composing a duplicate raises
`ValueError`.

### Creating stages with `stage`

Like Dask's `delayed`, the `stage` decorator/factory wraps a `fn(ctx)` into a
`Stage`, defaulting the name to the function's `__name__`:

```python
from vtdf import stage

@stage
def load(ctx):
    return ctx.path

@stage(name="train", ctx=cfg)
def fit(ctx):
    ...

pipeline = load >> fit
```

It can also be called directly: `stage(load, ctx=cfg)`.

### Artifacts

An `Artifact` is a remote resource — a GCS object, a BigQuery table — that sits in
the DAG as a source or sink, next to stages. It carries a `name`, a `uri`, and
optional `description` / `metadata`:

```python
from vtdf import Artifact

dataset = Artifact(
    name="training-dataset",
    uri="gs://your-bucket/data/train.csv",
    description="...",                  # optional
    metadata={"rows": 50000, "version": "v2.3"},  # optional
)
```

As a node, an artifact **yields itself**. So a consuming stage receives the
`Artifact` (read its `uri`), and a producing stage's own result is ignored — the
sink is the artifact:

```python
def load(dataset, ctx):
    return read_csv(dataset.uri)        # dataset is the Artifact

def train(rows, ctx):
    return fit(rows)

model = Artifact("model", "gs://your-bucket/models/m.pkl")

dag = dataset >> Stage("load", load) >> Stage("train", train) >> model
dag.run()  # -> the `model` Artifact (the sink)
```

Artifacts share the unique-name rule with stages and are hashed by identity.

### Async execution and concurrency

For DAGs of **remote jobs** — where each stage submits work and waits on the
result — use the async API so independent branches run concurrently instead of
one after another:

- `run_async()` — async counterpart of `run()`.
- `run_collect_async()` — async counterpart of `run_collect()`.

A stage's `fn` may be `async def`. Independent nodes (e.g. the `[a, b]` in
`[a, b] >> c`) are scheduled together and awaited concurrently:

```python
import asyncio
from vtdf import Stage

async def fetch(ctx):
    return await remote_job(ctx)        # awaits the remote result

a, b = Stage("a", fetch), Stage("b", fetch)
async def merge(ra, rb, ctx): return (ra, rb)
c = Stage("c", merge)

asyncio.run(([a, b] >> c).run_async())  # a and b run concurrently, then c
```

**Use `async def` everywhere for full concurrency.** Concurrency only kicks in
while a stage is `await`-ing. A plain `def` stage runs **inline** on the event
loop under `run_async()` — it blocks every other branch until it returns, so it
gains nothing and (worse) stalls stages that could otherwise overlap. `run_async`
emits a warning when it encounters a sync stage. Mixed graphs still produce
correct results; they just don't parallelize across the sync stages.

## How it works

### Data model

A `DAG` holds a single field, `deps: dict[Node, list[Node]]`, mapping each node
(a `Stage` or `Artifact`) to its predecessors in order (the nodes that must run
before it). That is the entire graph — there is no separate node list or mutable
build cursor.

The **frontier** (where the next `>>` attaches) is *derived* from the graph as
its sink nodes: nodes that appear as nobody's predecessor. Because the
frontier is computed, never stored, a `DAG` is an immutable value — composition
builds a new map and never mutates its operands, so a DAG can be reused or
composed further freely.

### Composition (`>>`)

Each operand — a node (`Stage` or `Artifact`), a list of nodes, or a `DAG` — is
first `_normalize`d into a deps map:

- node → `{node: ∅}`
- `list` of nodes → `{n: ∅ for n in list}` (parallel, no inter-dependencies)
- `DAG` → a fresh copy of its deps map

`_compose(left, right)` then:

1. Normalizes `left` into a new map `deps`.
2. Computes `frontier = sinks(deps)` — the nodes with no successors in `left`.
3. Merges every node of `right` into `deps`. Each **source** node of `right`
   (one with no predecessors of its own) gains the whole `frontier` as its
   predecessors; that is the edge that wires the two operands together.

This makes the operations associative over sub-DAGs: `(a >> b) >> (c >> d)`
attaches `c`'s sources to `b` (the sink of the left DAG), yielding the linear
chain `a → b → c → d`.

### Execution

Both run paths order the graph with the standard library's
[`graphlib.TopologicalSorter`](https://docs.python.org/3/library/graphlib.html)
over the `deps` map, which raises `graphlib.CycleError` (a `ValueError` whose
message contains "cycle") on a cyclic graph. Each node is invoked with its own
`ctx` if it carries one, otherwise the `ctx` passed to the run method.

- **Sync** (`run` / `run_collect`) walks `static_order()` and runs nodes one at a
  time, threading each result to its successors.
- **Async** (`run_async` / `run_collect_async`) drives the sorter's *active* API
  (`prepare` / `get_ready` / `done`): all currently-ready (independent) nodes are
  scheduled as `asyncio` tasks at once and awaited as they complete, so
  independent `async def` branches actually run concurrently. See
  [Async execution and concurrency](#async-execution-and-concurrency).

## Similar packages

- [`graphlib.TopologicalSorter`](https://docs.python.org/3/library/graphlib.html)
  — standard library (Python 3.9+). `vtdf` uses it directly as its graph engine:
  `static_order()` for the sync path and the `get_ready()`/`done()` active API for
  concurrent async scheduling. What `vtdf` adds on top is the `Stage`/`Artifact`
  wrappers, `>>` composition sugar, and the async run methods.
- [Dask `delayed`](https://docs.dask.org/en/stable/delayed.html) / dask graphs —
  builds a lazy compute graph and executes it (with real parallelism). Heavier,
  but conceptually the same "wrap functions, declare deps, run in order."
- Smaller PyPI packages in this exact niche:
  [`daglib`](https://pypi.org/project/daglib/),
  [`pydags`](https://pypi.org/project/pydags/),
  [`schedula`](https://pypi.org/project/schedula/),
  [`pyungo`](https://pypi.org/project/pyungo/),
  [`fn_graph`](https://pypi.org/project/fn-graph/) — most use decorators or
  `requires()`-style wiring rather than `>>`.

from dataclasses import dataclass

import pytest

from vtdf import DAG, Stage, stage


@dataclass(frozen=True)
class MyContext:
    learning_rate: float
    max_iter: int


def make_recorder():
    """Return (order, ctxs, factory) where factory(name) builds a recording Stage."""
    order = []
    ctxs = []

    def factory(name, ctx):
        def fn(c):
            order.append(name)
            ctxs.append(c)
        return Stage(name, fn, ctx)

    return order, ctxs, factory


def before(order, x, y):
    return order.index(x) < order.index(y)


def test_dag():
    ctx = MyContext(learning_rate=0.3, max_iter=100)

    def fn(name):
        return lambda c: None

    a = Stage("a", fn("a"), ctx)
    b = Stage("b", fn("b"), ctx)
    c = Stage("c", fn("c"), ctx)
    d = Stage("d", fn("d"), ctx)
    e = Stage("e", fn("e"), ctx)

    res = [a, b] >> c >> [d, e]
    res.run()


def test_fan_in_fan_out_order():
    ctx = MyContext(0.3, 100)
    order, _, mk = make_recorder()
    a, b, c, d, e = (mk(n, ctx) for n in "abcde")

    ([a, b] >> c >> [d, e]).run()

    assert set(order) == set("abcde")
    for x in "ab":
        assert before(order, x, "c")
    for y in "de":
        assert before(order, "c", y)


def test_linear_chain():
    ctx = MyContext(0.3, 100)
    order, _, mk = make_recorder()
    a, b, c = (mk(n, ctx) for n in "abc")

    (a >> b >> c).run()

    assert order == ["a", "b", "c"]


def test_ctx_passed_to_each_stage():
    ctx = MyContext(0.3, 100)
    order, ctxs, mk = make_recorder()
    a, b = mk("a", ctx), mk("b", ctx)

    (a >> b).run()

    assert ctxs == [ctx, ctx]


def test_each_stage_runs_once():
    ctx = MyContext(0.3, 100)
    order, _, mk = make_recorder()
    a, b, c, d = (mk(n, ctx) for n in "abcd")

    ([a, b] >> c >> [d]).run()

    assert sorted(order) == ["a", "b", "c", "d"]


def test_cycle_detected():
    ctx = MyContext(0.3, 100)
    _, _, mk = make_recorder()
    a, b = mk("a", ctx), mk("b", ctx)

    dag = DAG()
    dag.deps = {a: {b}, b: {a}}

    with pytest.raises(ValueError, match="cycle"):
        dag.run()


def test_composition_is_non_destructive():
    ctx = MyContext(0.3, 100)
    _, _, mk = make_recorder()
    a, b, c, d = (mk(n, ctx) for n in "abcd")

    base = a >> b
    left = base >> c
    right = base >> d

    assert set(base.deps) == {a, b}          # reused operand untouched
    assert set(left.deps) == {a, b, c}
    assert set(right.deps) == {a, b, d}


def test_sub_dags_compose():
    ctx = MyContext(0.3, 100)
    order, _, mk = make_recorder()
    a, b, c, d = (mk(n, ctx) for n in "abcd")

    ((a >> b) >> (c >> d)).run()

    assert order == ["a", "b", "c", "d"]


def test_run_ctx_with_stage_override():
    order, ctxs, mk = make_recorder()
    a = mk("a", None)                        # no ctx -> takes run() ctx
    b = mk("b", "override")                  # carries its own ctx

    (a >> b).run(ctx="run")

    assert ctxs == ["run", "override"]


def test_run_returns_results_by_name():
    a = Stage("a", lambda c: 1, None)
    b = Stage("b", lambda c: 2, None)

    out = (a >> b).run()

    assert out == {"a": 1, "b": 2}


def test_stage_decorator_bare():
    @stage
    def load(ctx):
        return ctx

    assert isinstance(load, Stage)
    assert load.name == "load"
    assert load.run("c") == "c"


def test_stage_decorator_parameterized():
    @stage(name="train", ctx="cfg")
    def fit(ctx):
        return ctx

    assert isinstance(fit, Stage)
    assert fit.name == "train"
    assert fit.run() == "cfg"


def test_stage_factory_and_composition():
    def load(ctx):
        return 1

    def train(ctx):
        return 2

    out = (stage(load) >> stage(train)).run()

    assert out == {"load": 1, "train": 2}

from dataclasses import dataclass

import pytest

from vtdf import DAG, Stage


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

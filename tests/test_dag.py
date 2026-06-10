from dataclasses import dataclass

import pytest

from vtdf import DAG, Stage, stage


@dataclass(frozen=True)
class MyContext:
    learning_rate: float
    max_iter: int


def make_recorder():
    """Return (order, calls, factory). Each recording Stage appends its name to
    `order`, stores {"preds": tuple_of_predecessor_results, "ctx": ctx} under its
    name in `calls`, and returns a tag `r_<name>` as its result."""
    order = []
    calls = {}

    def factory(name, ctx=None):
        def fn(*args):
            *preds, c = args
            order.append(name)
            calls[name] = {"preds": tuple(preds), "ctx": c}
            return f"r_{name}"

        return Stage(name, fn, ctx)

    return order, calls, factory


def before(order, x, y):
    return order.index(x) < order.index(y)


def test_dag():
    ctx = MyContext(learning_rate=0.3, max_iter=100)
    a, b, c, d, e = (Stage(n, lambda *args: None, ctx) for n in "abcde")

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


def test_results_thread_along_chain():
    order, calls, mk = make_recorder()
    a, b, c = (mk(n) for n in "abc")

    out = (a >> b >> c).run(ctx="CTX")

    assert calls["a"]["preds"] == ()  # source stage: only ctx
    assert calls["b"]["preds"] == ("r_a",)
    assert calls["c"]["preds"] == ("r_b",)
    assert out == "r_c"  # last job's result


def test_fan_in_passes_results_in_order():
    _, calls, mk = make_recorder()
    a, b, c = (mk(n) for n in "abc")

    ([a, b] >> c).run()

    assert calls["c"]["preds"] == ("r_a", "r_b")


def test_fan_out_shares_predecessor_result():
    _, calls, mk = make_recorder()
    a, d, e = (mk(n) for n in "ade")

    (a >> [d, e]).run()

    assert calls["d"]["preds"] == ("r_a",)
    assert calls["e"]["preds"] == ("r_a",)


def test_run_returns_single_sink_result():
    a = Stage("a", lambda ctx: 1)
    b = Stage("b", lambda ra, ctx: ra + 10)

    assert (a >> b).run() == 11


def test_run_returns_tuple_for_multiple_sinks():
    a = Stage("a", lambda ctx: 1)
    d = Stage("d", lambda ra, ctx: ra + 1)
    e = Stage("e", lambda ra, ctx: ra + 2)

    assert (a >> [d, e]).run() == (2, 3)


def test_ctx_passed_to_each_stage():
    ctx = MyContext(0.3, 100)
    order, calls, mk = make_recorder()
    a, b = mk("a", ctx), mk("b", ctx)

    (a >> b).run()

    assert calls["a"]["ctx"] == ctx
    assert calls["b"]["ctx"] == ctx


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
    dag.deps = {a: [b], b: [a]}

    with pytest.raises(ValueError, match="cycle"):
        dag.run()


def test_composition_is_non_destructive():
    ctx = MyContext(0.3, 100)
    _, _, mk = make_recorder()
    a, b, c, d = (mk(n, ctx) for n in "abcd")

    base = a >> b
    left = base >> c
    right = base >> d

    assert set(base.deps) == {a, b}  # reused operand untouched
    assert set(left.deps) == {a, b, c}
    assert set(right.deps) == {a, b, d}


def test_sub_dags_compose():
    ctx = MyContext(0.3, 100)
    order, _, mk = make_recorder()
    a, b, c, d = (mk(n, ctx) for n in "abcd")

    ((a >> b) >> (c >> d)).run()

    assert order == ["a", "b", "c", "d"]


def test_run_ctx_with_stage_override():
    _, calls, mk = make_recorder()
    a = mk("a", None)  # no ctx -> takes run() ctx
    b = mk("b", "override")  # carries its own ctx

    (a >> b).run(ctx="run")

    assert calls["a"]["ctx"] == "run"
    assert calls["b"]["ctx"] == "override"


def test_stage_decorator_bare():
    @stage
    def load(ctx):
        return ctx

    assert isinstance(load, Stage)
    assert load.name == "load"
    assert load.run(ctx="c") == "c"


def test_stage_decorator_parameterized():
    @stage(name="train", ctx="cfg")
    def fit(ctx):
        return ctx

    assert isinstance(fit, Stage)
    assert fit.name == "train"
    assert fit.run() == "cfg"


def test_static_stage_returns_value():
    assert Stage("const", 42).run() == 42


def test_static_stage_in_composition():
    out = (Stage("x", 1) >> Stage("y", [1, 2])).run()
    assert out == [1, 2]  # last job `y` is static


def test_stage_factory_static_value():
    s = stage(42, name="seed")
    assert isinstance(s, Stage)
    assert s.run() == 42


def test_stage_factory_static_value_requires_name():
    with pytest.raises(ValueError, match="explicit name"):
        stage(42)


def test_stage_factory_and_composition():
    def load(ctx):
        return 1

    def train(prev, ctx):
        return prev + 1

    out = (stage(load) >> stage(train)).run()

    assert out == 2

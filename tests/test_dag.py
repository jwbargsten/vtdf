from dataclasses import dataclass
from vtdf import Stage

@dataclass(frozen=True)
class MyContext:
    learning_rate: float
    max_iter: int

def fna(ctx: MyContext):
    print(f"got ctx {ctx=}")
def fnb(ctx: MyContext):
    print(f"got ctx {ctx=}")
def fnc(ctx: MyContext):
    print(f"got ctx {ctx=}")
def fnd(ctx: MyContext):
    print(f"got ctx {ctx=}")
def fne(ctx: MyContext):
    print(f"got ctx {ctx=}")

def test_dag():
    ctx = MyContext(learning_rate=0.3, max_iter=100)

    a = Stage("a", fna, ctx)
    b = Stage("b", fnb, ctx)
    c = Stage("c", fnc, ctx)
    d = Stage("d", fnd, ctx)
    e = Stage("e", fne, ctx)

    res1 = [a, b] >> c
    res2 = res1 >> [d, e]

    res2.run()



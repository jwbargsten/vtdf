# ctx is an immutable object (can be dict or custom type)

class Stage:
    def __init__(self, name, fn, ctx):
        self.name = name

    def __rshift__(self, other):
        return other

    def __rrshift__(self, other):
        return other

class DAG:
    def __init__(self):
        pass

    def __rshift__(self, other: "DAG | Stage | list[Stage]"):
        return other
    def __rrshift__(self, other: "DAG | Stage | list[Stage]"):
        return other

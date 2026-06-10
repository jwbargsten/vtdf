# ctx is an immutable object (can be dict or custom type)

class Stage:
    def __init__(self, name, fn, ctx):
        self.name = name

    def __rshift__(self, other):
        pass
    def __rrshift__(self, other):
        pass

class DAG:
    def __init__(self):

    def __rshift__(self, other):
        pass
    def __rrshift__(self, other):
        pass

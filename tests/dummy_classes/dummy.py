class DummyA:

    def __init__(self, state):
        self.state = state


class DummyB:

    def __init__(self, arg1: int, arg2: str, arg3: DummyA):
        self.state1 = arg1
        self.state2 = arg2
        self.state3 = arg3


class DummyC:

    def __init__(self):
        self.state = 42
        self.state2 = DummyA(42)

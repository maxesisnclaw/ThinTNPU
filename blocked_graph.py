from thin_backend.graph import CapturedForward


class FrozenGraph:
    def __init__(self, pilot, forward, example):
        pilot.validate()
        self.pilot = pilot
        self.graph = CapturedForward(forward, example)

    @property
    def outputs(self):
        return self.graph.outputs

    def replay(self):
        self.pilot.validate()
        return self.graph.replay()

    def __call__(self, value):
        self.pilot.validate()
        return self.graph(value)

    def close(self):
        self.graph.close()

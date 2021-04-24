
class NumericalDetails:
    def __init__(self, min=None, max=None, average=None):
        self.min = min
        self.max = max
        self.average = average

    def __str__(self) -> str:
        return f'<{self.min:.6} ... avg={self.average:.6} ... {self.max:.6}>'

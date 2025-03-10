from my_types import Vector

class ActivationFn:
    def __init__(self) -> None:
        pass

    def apply(self, x: Vector) -> Vector:
        pass

class ThresholdFn(ActivationFn):
    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold
    
    def apply(self, x: Vector) -> Vector:
        return x > self.threshold
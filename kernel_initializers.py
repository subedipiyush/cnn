import numpy as np

from my_types import Matrix

class KernelInitializer:
    def __init__(self, name: str) -> None:
        self.name = name

    def initialize(self, shape: tuple) -> Matrix:
        pass

class RandomNormal(KernelInitializer):
    def __init__(self) -> None:
        super().__init__("RandomNormal")
    
    def initialize(self, shape: tuple, mean: float = 0, std: float = 0.05) -> Matrix:
        return np.random.normal(loc=mean, scale=std, size=shape)
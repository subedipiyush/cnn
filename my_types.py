from __future__ import annotations
import numpy as np

class Vector(np.ndarray):
    def __new__(cls, input_array: list) -> np.ndarray:
        return np.asarray(input_array).view(cls)
    
    def length(self) -> float:
        return len(self)
    
    def dot(self, other: Vector) -> float:
        return np.dot(self, other)

class FloatVector(Vector):
    def __new__(cls, input_array: list[float]) -> np.ndarray:
        return np.asarray(input_array, dtype=np.float32).view(cls).reshape(1, len(input_array))

class Matrix(np.ndarray):
    def __new__(cls, input_array: list) -> np.ndarray:
        return np.asarray(input_array).view(cls)

    def dot(self, other: Vector) -> float:
        return np.dot(self, other)
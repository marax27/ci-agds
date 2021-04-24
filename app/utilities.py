from typing import Callable, List, TypeVar

T = TypeVar('T')


class NumericalDetails:
    def __init__(self, min=None, max=None, average=None):
        self.min = min
        self.max = max
        self.average = average

    def __str__(self) -> str:
        return f'<{self.min:.6} ... avg={self.average:.6} ... {self.max:.6}>'


def binary_search(collection: List[T], value: float, mapper: Callable[[T], float]=None) -> float:
    if mapper is None:
        mapper = lambda x: x
    low, high = 0, len(collection) - 1

    while low <= high:
        mid = (low + high) // 2
        if mapper(collection[mid]) > value: high = mid - 1
        elif mapper(collection[mid]) < value: low = mid + 1
        else: return mid

    if low >= len(collection):  return high
    if high < 0:  return low

    vlow = abs(mapper(collection[low]) - value)
    vhigh = abs(mapper(collection[high]) - value)
    return low if vlow < vhigh else high
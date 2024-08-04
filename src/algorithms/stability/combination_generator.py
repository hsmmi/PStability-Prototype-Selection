import itertools
import math
from typing import Iterator, Tuple
import numpy as np


class CombinationGenerator:
    """
    A class to generate and provide iterators over combinations of selected indices.
    """

    def __init__(self):
        self._n: int = 0
        self._p: int = 0
        self._total_combinations: int = 0

    def configure(self, n: int, p: int) -> "CombinationGenerator":
        """
        Configures the parameters for generating combinations of selected indices.

        Args:
            n (int): The total number of indices.
            p (int): The number of indices in each combination.

        Returns:
            CombinationGenerator: Returns the instance itself.
        """
        if n <= 0:
            raise ValueError("n must be a positive integer.")
        if p <= 0:
            raise ValueError("p must be a positive integer.")
        if p > n:
            raise ValueError("p cannot be greater than the number of indices (n).")

        self._n = n
        self._p = p
        self._total_combinations = math.comb(n, p)
        return self

    def __iter__(self) -> Iterator[Tuple[np.ndarray]]:
        """
        Returns an iterator over the combinations of selected indices.

        Returns:
            Iterator[Tuple[np.ndarray]]: An iterator over the combinations of selected indices.
        """
        return _CombinationIterator(self._n, self._p)

    def __len__(self) -> int:
        """
        Returns the total number of combinations.

        Returns:
            int: Total number of combinations.
        """
        return self._total_combinations


class _CombinationIterator:
    """
    An iterator to iterate over combinations of selected indices.
    """

    def __init__(self, n: int, p: int):
        self._comb_iterator = iter(itertools.combinations(range(n), p))

    def __iter__(self) -> "_CombinationIterator":
        """
        Returns itself as an iterator.

        Returns:
            _CombinationIterator: Returns itself as an iterator.
        """
        return self

    def __next__(self) -> Tuple[np.ndarray]:
        """
        Returns the next combination of selected indices.

        Returns:
            Tuple[np.ndarray]: The next combination of selected indices as a numpy array.

        Raises:
            StopIteration: If no more combinations are available.
        """
        try:
            selected = next(self._comb_iterator)  # Get the next combination of indices
            return np.array(selected)  # Return the selected indices as a numpy array
        except StopIteration:
            raise StopIteration  # Raise StopIteration when there are no more combinations

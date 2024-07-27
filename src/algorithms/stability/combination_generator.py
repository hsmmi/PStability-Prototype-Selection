import itertools
import math
from typing import Iterator, Tuple
from tqdm import tqdm
import numpy as np


class CombinationGenerator:
    """
    A class to generate and iterate over selected indices and their remaining indices.
    """

    def __init__(self):
        self.n: int = 0
        self.p: int = 0
        self.total_combinations: int = 0
        self._comb_iterator: Iterator[Tuple[int, ...]] = iter([])

    def set_params(self, n: int, p: int) -> "CombinationGenerator":
        """
        Sets the parameters for generating selected indices and remaining indices.

        Args:
            n (int): The total number of indices.
            p (int): The number of indices in the selected set.

        Returns:
            CombinationGenerator: Returns the instance itself.
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer.")
        if not isinstance(p, int) or p <= 0:
            raise ValueError("p must be a positive integer.")
        if p > n:
            raise ValueError("p cannot be greater than the number of indices (n).")

        self.n = n
        self.p = p
        self.total_combinations = math.comb(n, p)
        # TODO: Add shuffle parameter to shuffle the combinations
        self._comb_iterator = itertools.combinations(range(n), p)
        return self

    def __iter__(self) -> Iterator[Tuple[np.ndarray]]:
        """
        Makes the CombinationGenerator an iterable object.

        Returns:
            Iterator[Tuple[np.ndarray]]: An iterator over selected indices.
        """
        # Re-initialize the combination iterator for iteration
        self._comb_iterator = itertools.combinations(range(self.n), self.p)
        return self

    def __next__(self) -> Tuple[np.ndarray]:
        """
        Returns the next selected indices in the iteration.

        Returns:
            Tuple[np.ndarray]: The next selected indices as a numpy array.

        Raises:
            StopIteration: If no more combinations are available.
        """
        try:
            selected = next(self._comb_iterator)  # Get the next combination of indices
            return np.array(selected)  # Return the selected indices as a numpy array
        except StopIteration:
            raise StopIteration  # Raise StopIteration when there are no more combinations

    def __len__(self) -> int:
        """
        Returns the total number of combinations.

        Returns:
            int: Total number of combinations.
        """
        return self.total_combinations

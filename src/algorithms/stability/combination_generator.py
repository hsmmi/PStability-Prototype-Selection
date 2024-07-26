import itertools
import math
from typing import Iterator, Tuple
from tqdm import tqdm
import numpy as np


class CombinationGenerator:
    """
    A class to generate combinations of a given length from a dataset.

    Attributes:
        n (int): The total number of elements in the dataset.
        p (int): The number of elements in each combination.
        total_combinations (int): The total number of combinations possible.
        generator_combinations (Iterator[Tuple[int, ...]]): The generator for combinations.
    """

    def __init__(self, n: int, p: int):
        """
        Initializes the CombinationGenerator with the dataset length and combination length.

        Args:
            n (int): The length of the dataset.
            p (int): The number of elements in each combination.
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer.")
        if not isinstance(p, int) or p <= 0:
            raise ValueError("p must be a positive integer.")
        if p > n:
            raise ValueError("p cannot be greater than the number of elements (n).")

        self.n: int = n
        self.p: int = p
        self.total_combinations: int = math.comb(n, p)
        self.generator_combinations: Iterator[Tuple[int, ...]] = itertools.combinations(
            range(n), p
        )

    def generate_combinations(self) -> Iterator[np.ndarray]:
        """
        Generates combinations with a progress bar.

        Yields:
            np.ndarray: A numpy array containing a combination of indices.
        """
        for combination in tqdm(
            self.generator_combinations,
            total=self.total_combinations,
            desc=f"C({self.n}, {self.p})",
            leave=False,
        ):
            yield np.array(combination)

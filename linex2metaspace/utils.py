from itertools import product
from typing import List


def flatten(l: List):
    return [item for sublist in l for item in sublist]


def list_product(l1: List, l2: List):
    return list(product(l1, l2))

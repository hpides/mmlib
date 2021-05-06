from typing import Dict

import torch
from torch import Tensor

from util.hash import tensor_hash


class WeightDictMerkleTreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    @property
    def hash_value(self):
        if isinstance(str, self.value):
            return self.value
        elif isinstance(torch.tensor, self.value):
            return tensor_hash(self.value)


class WeightDictMerkleTree:

    def __init__(self, weight_dict: Dict[str, Tensor]):
        pass

    def to_dict(self) -> dict:
        pass

    @classmethod
    def from_dict(cls, hash_info_dict):
        pass

    def __eq__(self, other):
        # to implement this method we only have to compare the root hash
        return False

    def __hash__(self):
        # here we can just return the root hash
        return hash(self)

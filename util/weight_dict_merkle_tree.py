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
        self.root = self._build_tree(weight_dict)
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

    def _build_tree(self, weight_dict):

        # unlike a normal tree for a merkel tree we start with the leaves
        leaves = []
        for key, value in weight_dict.items():
            weight_hash = tensor_hash(value)
            leaves.append(WeightDictMerkleTreeNode(weight_hash))

        # after having all leave nodes we add layer by layer on top until we have only one single node left
        # the single node left is then the root node
        current_layer = leaves
        while len(current_layer) > 1:
            current_layer = self._build_next_layer(current_layer)

        assert len(current_layer) == 1
        return current_layer[0]

    def _build_next_layer(self, current_layer):
        pass

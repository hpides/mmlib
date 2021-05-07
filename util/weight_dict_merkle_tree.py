import math
from typing import Dict

import torch
from torch import Tensor

from util.hash import tensor_hash, hash_string


class WeightDictMerkleTreeNode:
    def __init__(self, value, left=None, right=None):
        self._value = value
        self.left = left
        self.right = right

    @property
    def hash_value(self):
        if isinstance(str, self._value):
            return self._value
        elif isinstance(torch.tensor, self._value):
            return tensor_hash(self._value)


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

        # as soon as we build all leave nodes we have to build the upper layers
        # to build a balance tree we start taking 2 nodes from the beginning of the leave nodes an build a new node
        # as soon as we have only 2^x nodes left we can just combine all -> because then we can build a balanced tree
        num_leaves = len(leaves)
        current_layer = []
        while not math.log(num_leaves, 2).is_integer():
            left = leaves.pop(0)
            right = leaves.pop(0)
            value = hash_string(left.hash_value + right.hash_value)
            node = WeightDictMerkleTreeNode(value=value, left=left, right=right)
            current_layer.append(node)
            num_leaves -= 1

        # now we know that the current layer has a number fo elements equal to 2^x
        # we combine nodes as long as in the current layer there is only one node -> the root
        current_layer = leaves
        while len(current_layer) > 1:
            current_layer = self._build_next_layer(current_layer)

        assert len(current_layer) == 1
        return current_layer[0]

    def _build_next_layer(self, current_layer):
        # since a merkle tree is a binary tree, for two nodes each we build a new parent node
        # if the number of current nodes is odd this node is lifted one layer up
        new_layer = []
        for i in range(0, len(current_layer), 2):
            left = current_layer[i]
            right = current_layer[i + 1]
            value = hash_string(left + right)
            node = WeightDictMerkleTreeNode(value=value, left=left, right=right)
            new_layer.append(node)

        return new_layer

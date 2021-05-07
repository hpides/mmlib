import math
from typing import Dict

import torch
from torch import Tensor, tensor

from util.hash import tensor_hash, hash_string

HASH_VALUE = 'hash_value'
LEFT = 'left'
RIGHT = 'right'


class WeightDictMerkleTreeNode:
    def __init__(self, value, left=None, right=None):
        self._value = value
        self.left = left
        self.right = right

    @property
    def hash_value(self):
        if isinstance(self._value, str):
            return self._value
        elif isinstance(self._value, tensor):
            return tensor_hash(self._value)

    def to_dict(self):
        result = {HASH_VALUE: self.hash_value}
        if self.left:
            result[LEFT] = self.hash_value
        if self.right:
            result[RIGHT] = self.hash_value

        return result


def to_node(hash_info_dict):
    if HASH_VALUE not in hash_info_dict:
        return None
    else:
        value = hash_info_dict[HASH_VALUE]
        left = None
        right = None
        if LEFT in hash_info_dict:
            left = to_node(hash_info_dict[LEFT])
        if RIGHT in hash_info_dict:
            right = to_node(hash_info_dict[RIGHT])

        return WeightDictMerkleTreeNode(value=value, left=left, right=right)


class WeightDictMerkleTree:

    def __init__(self, weight_dict: Dict[str, Tensor] = None):
        self.root = None
        if weight_dict:
            self.root = self._build_tree(weight_dict)

    def to_dict(self) -> dict:
        return self.root.to_dict()

    @classmethod
    def from_dict(cls, hash_info_dict):
        tree = WeightDictMerkleTree()
        tree.root = to_node(hash_info_dict)
        return tree

    def __eq__(self, other):
        return self.root.hash_value == other.root.hash_value

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

        current_layer = leaves + current_layer
        # now we know that the current layer has a number fo elements equal to 2^x
        # we combine nodes as long as in the current layer there is only one node -> the root
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
            value = hash_string(left.hash_value + right.hash_value)
            node = WeightDictMerkleTreeNode(value=value, left=left, right=right)
            new_layer.append(node)

        return new_layer

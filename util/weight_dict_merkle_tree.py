import math
from typing import Dict

from torch import Tensor

from util.hash import tensor_hash, hash_string

HASH_VALUE = 'hash_value'
LEFT = 'left'
RIGHT = 'right'
LAYER_KEY = 'layer_key'
LAYER_WEIGHT_HASH = 'layer_weight_hash'


class WeightDictMerkleTreeNode:
    def __init__(self, hash_value=None, left=None, right=None, layer_key=None, layer_weights_hash=None):
        if hash_value is None:
            assert layer_key and layer_weights_hash
            hash_value = hash_string(layer_weights_hash + layer_key)
        self.hash_value = hash_value
        self.left = left
        self.right = right
        self.layer_key = layer_key
        self.layer_weights_hash = layer_weights_hash

    def to_dict(self):
        result = {HASH_VALUE: self.hash_value}
        if self.layer_key:
            result[LAYER_KEY] = self.layer_key
            result[LAYER_WEIGHT_HASH] = self.layer_weights_hash
            assert self.left is None and self.right is None, 'only a leave has a layer key'
            assert self.layer_weights_hash is not None, 'if layer key is given weight hash also must be givens'
        if self.left:
            result[LEFT] = self.left.to_dict()
        if self.right:
            result[RIGHT] = self.right.to_dict()

        return result

    def check_integrity(self):
        if self.left and self.right:
            return self.left.check_integrity() and \
                   self.right.check_integrity() and \
                   self.hash_value == hash_string(self.left.hash_value + self.right.hash_value)
        else:
            assert self.left is None and self.right is None
            return self.hash_value == hash_string(self.layer_weights_hash + self.layer_key)


def to_node(hash_info_dict):
    value = hash_info_dict[HASH_VALUE]
    layer_key = hash_info_dict[LAYER_KEY] if LAYER_KEY in hash_info_dict else None
    layer_weight_hash = hash_info_dict[LAYER_WEIGHT_HASH] if LAYER_WEIGHT_HASH in hash_info_dict else None
    left = None
    right = None

    if LEFT in hash_info_dict:
        left = to_node(hash_info_dict[LEFT])
    if RIGHT in hash_info_dict:
        right = to_node(hash_info_dict[RIGHT])

    return WeightDictMerkleTreeNode(
        hash_value=value, left=left, right=right, layer_key=layer_key, layer_weights_hash=layer_weight_hash)


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
        assert tree.root.check_integrity()  # TODO maybe throw parsing error
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
            leaves.append(WeightDictMerkleTreeNode(layer_key=key, layer_weights_hash=tensor_hash(value)))

        # as soon as we build all leave nodes we have to build the upper layers
        # to build a balance tree we start taking 2 nodes from the beginning of the leave nodes an build a new node
        # as soon as we have only 2^x nodes left we can just combine all -> because then we can build a balanced tree
        num_leaves = len(leaves)
        current_layer = []
        while not math.log(num_leaves, 2).is_integer():
            left = leaves.pop(0)
            right = leaves.pop(0)
            value = hash_string(left.hash_value + right.hash_value)
            node = WeightDictMerkleTreeNode(hash_value=value, left=left, right=right)
            current_layer.append(node)
            num_leaves -= 1

        current_layer += leaves
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
            node = WeightDictMerkleTreeNode(hash_value=value, left=left, right=right)
            new_layer.append(node)

        return new_layer

import math
from typing import Dict

from torch import Tensor

from util.hash import tensor_hash, hash_string

OTHER = 'other'

THIS = 'this'

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

    @property
    def is_leave(self):
        return self.left is None and self.right is None

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

    def __eq__(self, other):
        return self.hash_value == other.hash_value

    def __hash__(self):
        return hash(self.hash_value)

    def diff_layers(self, other):
        diff_layers = {THIS: set(), OTHER: set()}
        if self == other:
            return diff_layers
        elif other is None:
            diff_layers[THIS] = diff_layers[THIS].union(self.get_all_leaves())
            return diff_layers
        else:
            diff_layers = {THIS: set(), OTHER: set()}
            if self.is_leave or other.is_leave:
                this_leaves = self.get_all_leaves()
                diff_layers[THIS] = diff_layers[THIS].union(this_leaves)
                other_leaves = other.get_all_leaves()
                diff_layers[OTHER] = diff_layers[OTHER].union(other_leaves)
            else:
                if self.left:
                    left_diff = self.left.diff_layers(other.left)
                    diff_layers[THIS] = diff_layers[THIS].union(left_diff[THIS])
                    diff_layers[OTHER] = diff_layers[OTHER].union(left_diff[OTHER])
                if self.right:
                    right_diff = self.right.diff_layers(other.right)
                    diff_layers[THIS] = diff_layers[THIS].union(right_diff[THIS])
                    diff_layers[OTHER] = diff_layers[OTHER].union(right_diff[OTHER])

            # NOTE there must be a better solution, but for now we go with this one
            tmp = diff_layers[THIS].copy()
            diff_layers[THIS] = diff_layers[THIS].difference(diff_layers[OTHER])
            diff_layers[OTHER] = diff_layers[OTHER].difference(tmp)
            return diff_layers

    def get_all_leaves(self):
        leaves = set()
        if self.is_leave:
            leaves.add(self)
            return leaves
        else:
            if self.left:
                leaves = leaves.union(self.left.get_all_leaves())
            if self.right:
                leaves = leaves.union(self.right.get_all_leaves())
            return leaves


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


def _index_by_layer_key(diff):
    result = {}
    for node in diff:
        result[node.layer_key] = node
    return result


def _remove_keys(dictionary, keys):
    for key in keys:
        del dictionary[key]


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

    def diff_layers(self, other):
        diff_nodes = {}

        diff = self.root.diff_layers(other.root)
        this_diff = _index_by_layer_key(diff[THIS])
        other_diff = _index_by_layer_key(diff[OTHER])

        diff_weights = set(this_diff.keys()).intersection(set(other_diff.keys()))

        _remove_keys(this_diff, diff_weights)
        _remove_keys(other_diff, diff_weights)

        diff_nodes[THIS] = set(this_diff.keys())
        diff_nodes[OTHER] = set(other_diff.keys())



        return diff_weights, diff_nodes



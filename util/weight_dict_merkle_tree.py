import math

from util.hash import hash_string, tensor_hash

OTHER = 'other'

THIS = 'this'

HASH_VALUE = 'hash_value'
LEFT = 'left'
RIGHT = 'right'
LAYER_KEY = 'layer_key'
LAYER_WEIGHT_HASH = 'layer_weight_hash'


class WeightDictMerkleTree:
    def __init__(self, hash_value=None, left=None, right=None, layer_key=None, layer_weights_hash=None):
        """
        :param hash_value: Tha hash value that the node should represent
        (if not given layer_key and layer_weights_hash must be given)
        :param left: The left subtree
        :param right: The right subtree
        :param layer_key: A string representing the key of the layer from the state dict
        :param layer_weights_hash: The hash of the weight dict that was stored under the layer_key in the state dict
        """
        if hash_value is None:
            assert layer_key and layer_weights_hash
            hash_value = hash_string(layer_weights_hash + layer_key)
        self.hash_value = hash_value
        self.left = left
        self.right = right
        self.layer_key = layer_key
        self.layer_weights_hash = layer_weights_hash

    @classmethod
    def from_state_dict(cls, state_dict):
        """
        Creates a merkle tree from a pytorch model state dict
        :param state_dict: The state dict to create the tree form.
        :return: The root of the created tree.
        """
        # unlike a normal tree for a merkel tree we start with the leaves
        leaves = []
        for key, value in state_dict.items():
            leaves.append(cls(layer_key=key, layer_weights_hash=tensor_hash(value)))

        # as soon as we build all leave nodes we have to build the upper layers
        # to build a balance tree we start taking 2 nodes from the beginning of the leave nodes an build a new node
        # as soon as we have only 2^x nodes left we can just combine all -> because then we can build a balanced tree
        num_leaves = len(leaves)
        current_layer = []
        while not math.log(num_leaves, 2).is_integer():
            left = leaves.pop(0)
            right = leaves.pop(0)
            value = hash_string(left.hash_value + right.hash_value)
            node = cls(hash_value=value, left=left, right=right)
            current_layer.append(node)
            num_leaves -= 1

        current_layer += leaves
        # now we know that the current layer has a number fo elements equal to 2^x
        # we combine nodes as long as in the current layer there is only one node -> the root
        while len(current_layer) > 1:
            current_layer = cls._build_next_layer(current_layer)

        assert len(current_layer) == 1
        return current_layer[0]

    @classmethod
    def _build_next_layer(cls, current_layer):
        # since a merkle tree is a binary tree, for two nodes each we build a new parent node
        # if the number of current nodes is odd this node is lifted one layer up
        new_layer = []
        for i in range(0, len(current_layer), 2):
            left = current_layer[i]
            right = current_layer[i + 1]
            value = hash_string(left.hash_value + right.hash_value)
            node = cls(hash_value=value, left=left, right=right)
            new_layer.append(node)

        return new_layer

    @classmethod
    def from_python_dict(cls, python_dict):
        """
        Reads a merkel tree that was serialized to a python dictionary.
        :param python_dict: The merkel tree represented as a python dict.
        :return: The root of the created tree.

        """
        value = python_dict[HASH_VALUE]
        layer_key = python_dict[LAYER_KEY] if LAYER_KEY in python_dict else None
        layer_weight_hash = python_dict[LAYER_WEIGHT_HASH] if LAYER_WEIGHT_HASH in python_dict else None
        left = None
        right = None

        if LEFT in python_dict:
            left = cls.from_python_dict(python_dict[LEFT])
        if RIGHT in python_dict:
            right = cls.from_python_dict(python_dict[RIGHT])

        root = cls(hash_value=value, left=left, right=right, layer_key=layer_key, layer_weights_hash=layer_weight_hash)
        assert root.check_integrity()  # TODO maybe throw parsing error
        return root

    @property
    def is_leave(self):
        """Checks if the given tree is a leave"""
        return self.left is None and self.right is None

    def __eq__(self, other):
        return self.hash_value == other.hash_value

    def __hash__(self):
        return hash(self.hash_value)

    def to_python_dict(self):
        """Serializes the tree object into a python dictionary"""
        result = {HASH_VALUE: self.hash_value}
        if self.layer_key:
            result[LAYER_KEY] = self.layer_key
            result[LAYER_WEIGHT_HASH] = self.layer_weights_hash
            assert self.left is None and self.right is None, 'only a leave has a layer key'
            assert self.layer_weights_hash is not None, 'if layer key is given weight hash also must be givens'
        if self.left:
            result[LEFT] = self.left.to_python_dict()
        if self.right:
            result[RIGHT] = self.right.to_python_dict()

        return result

    def check_integrity(self):
        """Checks if the given tree is integer: If every node holds the hash of both of its children"""
        if self.left and self.right:
            return self.left.check_integrity() and \
                   self.right.check_integrity() and \
                   self.hash_value == hash_string(self.left.hash_value + self.right.hash_value)
        else:
            assert self.left is None and self.right is None
            return self.hash_value == hash_string(self.layer_weights_hash + self.layer_key)

    def _diff_layers(self, other):
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
                    left_diff = self.left._diff_layers(other.left)
                    diff_layers[THIS] = diff_layers[THIS].union(left_diff[THIS])
                    diff_layers[OTHER] = diff_layers[OTHER].union(left_diff[OTHER])
                if self.right:
                    right_diff = self.right._diff_layers(other.right)
                    diff_layers[THIS] = diff_layers[THIS].union(right_diff[THIS])
                    diff_layers[OTHER] = diff_layers[OTHER].union(right_diff[OTHER])

            # NOTE there must be a better solution, but for now we go with this one
            tmp = diff_layers[THIS].copy()
            diff_layers[THIS] = diff_layers[THIS].difference(diff_layers[OTHER])
            diff_layers[OTHER] = diff_layers[OTHER].difference(tmp)

            return diff_layers

    def diff(self, other):
        """
        Compares the represented merkle tree to another given merkle tree.
        :param other: The othe rmekrle tree to compare to.
        :return: Returns a set of different weights found and a set of different nodes found.
        """
        diff_layers = self._diff_layers(other)

        diff_nodes = {}

        this_diff = _index_by_layer_key(diff_layers[THIS])
        other_diff = _index_by_layer_key(diff_layers[OTHER])

        diff_weights = set(this_diff.keys()).intersection(set(other_diff.keys()))

        _remove_keys(this_diff, diff_weights)
        _remove_keys(other_diff, diff_weights)

        diff_nodes[THIS] = set(this_diff.keys())
        diff_nodes[OTHER] = set(other_diff.keys())

        return diff_weights, diff_nodes

    def get_all_leaves(self):
        """
        Gives all leaves of the merkle tree back as a set of leave nodes.
        """
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


def _index_by_layer_key(diff):
    result = {}
    for node in diff:
        result[node.layer_key] = node
    return result


def _remove_keys(dictionary, keys):
    for key in keys:
        del dictionary[key]

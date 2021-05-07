import unittest

from torchvision.models import mobilenet_v2

from util.weight_dict_merkle_tree import WeightDictMerkleTree


class TestWeightDictMerkleTree(unittest.TestCase):

    def test_mobile_net_to_dict_from_dict(self):
        mobilenet = mobilenet_v2(pretrained=True)
        state_dict = mobilenet.state_dict()

        tree = WeightDictMerkleTree(state_dict)
        tree_to_dict = tree.to_dict()
        new_tree = WeightDictMerkleTree.from_dict(tree_to_dict)

        self.assertEqual(tree, new_tree)

    def test_two_dict_same_tree(self):
        mobilenet = mobilenet_v2(pretrained=True)
        state_dict = mobilenet.state_dict()

        tree1 = WeightDictMerkleTree(state_dict)
        tree2 = WeightDictMerkleTree(state_dict)
        self.assertEqual(tree1, tree2)

    def test_diff_dict_diff_tree(self):
        mobilenet = mobilenet_v2(pretrained=True)
        state_dict = mobilenet.state_dict()

        tree1 = WeightDictMerkleTree(state_dict)

        last_key = list(state_dict.keys())[-1]
        del state_dict[last_key]
        tree2 = WeightDictMerkleTree(state_dict)

        self.assertNotEqual(tree1, tree2)








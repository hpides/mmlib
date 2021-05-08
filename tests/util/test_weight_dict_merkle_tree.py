import unittest

import torch
from torchvision.models import mobilenet_v2

from util.weight_dict_merkle_tree import WeightDictMerkleTree, LEFT, LAYER_KEY, THIS, OTHER


class TestWeightDictMerkleTree(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.dummy_dict = {
            'layer1': torch.rand([10, 10]),
            'layer2': torch.rand([10, 10]),
            'layer3': torch.rand([10, 10]),
            'layer4': torch.rand([10, 10]),
            'layer5': torch.rand([10, 10])
        }

        self.dummy_dict2 = self.dummy_dict.copy()
        self.dummy_dict2['layer4'] = torch.rand([10, 10])
        self.dummy_dict2['layer5'] = torch.rand([10, 10])

        self.dummy_dict3 = self.dummy_dict.copy()
        self.dummy_dict3['layer6'] = torch.rand([10, 10])
        self.dummy_dict3['layer7'] = torch.rand([10, 10])

    def test_dummy_to_dict_to_tree(self):
        self._test_to_dict_to_tree(self.dummy_dict)

    def test_mobilenet_to_dict_to_tree(self):
        mobilenet = mobilenet_v2(pretrained=True)
        state_dict = mobilenet.state_dict()
        self._test_to_dict_to_tree(state_dict)

    def _test_to_dict_to_tree(self, _dict):
        tree1 = WeightDictMerkleTree(_dict)
        dict1 = tree1.to_dict()

        tree2 = WeightDictMerkleTree.from_dict(dict1)
        dict2 = tree2.to_dict()

        tree3 = WeightDictMerkleTree.from_dict(dict2)
        dict3 = tree3.to_dict()

        self.assertEqual(tree1, tree2)
        self.assertEqual(tree2, tree3)
        self.assertEqual(dict1, dict2)
        self.assertEqual(dict2, dict3)

    def test_dummy_two_dict_same_tree(self):
        self._test_two_dict_same_tree(self.dummy_dict)

    def test_mobilenet_two_dict_same_tree(self):
        mobilenet = mobilenet_v2(pretrained=True)
        state_dict = mobilenet.state_dict()
        self._test_two_dict_same_tree(state_dict)

    def _test_two_dict_same_tree(self, _dict):
        tree1 = WeightDictMerkleTree(_dict)
        tree2 = WeightDictMerkleTree(_dict)
        self.assertEqual(tree1, tree2)

    def test_diff_dict_diff_tree(self):
        mobilenet = mobilenet_v2(pretrained=True)
        state_dict = mobilenet.state_dict()

        tree1 = WeightDictMerkleTree(state_dict)

        last_key = list(state_dict.keys())[-1]
        del state_dict[last_key]
        tree2 = WeightDictMerkleTree(state_dict)

        self.assertNotEqual(tree1, tree2)

    def test_diff_last_layer_name(self):
        state_dict = self.dummy_dict

        tree1 = WeightDictMerkleTree(state_dict)

        last_key = list(state_dict.keys())[-1]
        tmp = state_dict[last_key]
        del state_dict[last_key]
        state_dict[last_key + 'x'] = tmp
        tree2 = WeightDictMerkleTree(state_dict)

        self.assertNotEqual(tree1, tree2)

    def test_not_integer_dict(self):
        tree1 = WeightDictMerkleTree(self.dummy_dict)
        dict1 = tree1.to_dict()

        dict1[LEFT][LEFT][LEFT][LAYER_KEY] += 'x'
        with self.assertRaises(Exception):
            WeightDictMerkleTree.from_dict(dict1)

    def test_diff_same_trees(self):
        tree1 = WeightDictMerkleTree(self.dummy_dict)
        tree2 = WeightDictMerkleTree(self.dummy_dict)

        diff = tree1.diff_layers(tree2)
        expected = {THIS: set(), OTHER: set()}
        self.assertEqual(diff, expected)

    def test_diff_same_architecture_diff_weights(self):
        tree1 = WeightDictMerkleTree(self.dummy_dict)
        tree2 = WeightDictMerkleTree(self.dummy_dict2)

        diff = tree1.diff_layers(tree2)
        self.assertEqual(len(diff[THIS]), 2)
        self.assertEqual(len(diff[OTHER]), 2)


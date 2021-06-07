import unittest

import torch
from torchvision.models import mobilenet_v2

from tests.example_files.mynets.resnet152 import resnet152
from mmlib.util.weight_dict_merkle_tree import WeightDictMerkleTree, LEFT, LAYER_KEY, THIS, OTHER

LAYER_1 = 'layer1'
LAYER_2 = 'layer2'
LAYER_3 = 'layer3'
LAYER_4 = 'layer4'
LAYER_5 = 'layer5'
LAYER_6 = 'layer6'
LAYER_7 = 'layer7'


class TestWeightDictMerkleTree(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.dummy_dict = {
            LAYER_1: torch.rand([10, 10]),
            LAYER_2: torch.rand([10, 10]),
            LAYER_3: torch.rand([10, 10]),
            LAYER_4: torch.rand([10, 10]),
            LAYER_5: torch.rand([10, 10])
        }

        self.dummy_dict2 = self.dummy_dict.copy()
        self.dummy_dict2[LAYER_4] = torch.rand([10, 10])
        self.dummy_dict2[LAYER_5] = torch.rand([10, 10])

        self.dummy_dict3 = self.dummy_dict.copy()
        self.dummy_dict3[LAYER_6] = torch.rand([10, 10])
        self.dummy_dict3[LAYER_7] = torch.rand([10, 10])

        self.dummy_dict4 = self.dummy_dict.copy()
        self.dummy_dict4[LAYER_1] = torch.rand([10, 10])
        self.dummy_dict4[LAYER_6] = torch.rand([10, 10])
        self.dummy_dict4[LAYER_7] = torch.rand([10, 10])

    def test_dummy_to_dict_to_tree(self):
        self._test_to_dict_to_tree(self.dummy_dict)

    def test_mobilenet_to_dict_to_tree(self):
        mobilenet = mobilenet_v2(pretrained=True)
        state_dict = mobilenet.state_dict()
        self._test_to_dict_to_tree(state_dict)

    def test_resnet_to_dict_to_tree(self):
        resnet = resnet152(pretrained=True)
        state_dict = resnet.state_dict()
        self._test_to_dict_to_tree(state_dict)

    def _test_to_dict_to_tree(self, _dict):
        tree1 = WeightDictMerkleTree.from_state_dict(_dict)
        dict1 = tree1.to_python_dict()

        tree2 = WeightDictMerkleTree.from_python_dict(dict1)
        dict2 = tree2.to_python_dict()

        tree3 = WeightDictMerkleTree.from_python_dict(dict2)
        dict3 = tree3.to_python_dict()

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

    def test_resnet152_two_dict_same_tree(self):
        resnet = resnet152(pretrained=True)
        state_dict = resnet.state_dict()
        self._test_two_dict_same_tree(state_dict)

    def _test_two_dict_same_tree(self, _dict):
        tree1 = WeightDictMerkleTree.from_state_dict(_dict)
        tree2 = WeightDictMerkleTree.from_state_dict(_dict)
        self.assertEqual(tree1, tree2)

    def test_diff_dict_diff_tree(self):
        mobilenet = mobilenet_v2(pretrained=True)
        state_dict = mobilenet.state_dict()

        tree1 = WeightDictMerkleTree(state_dict)

        last_key = list(state_dict.keys())[-1]
        del state_dict[last_key]
        tree2 = WeightDictMerkleTree.from_state_dict(state_dict)

        self.assertNotEqual(tree1, tree2)

    def test_diff_last_layer_name(self):
        state_dict = self.dummy_dict

        tree1 = WeightDictMerkleTree.from_state_dict(state_dict)

        last_key = list(state_dict.keys())[-1]
        tmp = state_dict[last_key]
        del state_dict[last_key]
        state_dict[last_key + 'x'] = tmp
        tree2 = WeightDictMerkleTree(state_dict)

        self.assertNotEqual(tree1, tree2)

    def test_not_integer_dict(self):
        tree1 = WeightDictMerkleTree.from_state_dict(self.dummy_dict)
        dict1 = tree1.to_python_dict()

        dict1[LEFT][LEFT][LEFT][LAYER_KEY] += 'x'
        with self.assertRaises(Exception):
            WeightDictMerkleTree.from_python_dict(dict1)

    def test_diff_same_trees(self):
        tree1 = WeightDictMerkleTree.from_state_dict(self.dummy_dict)
        tree2 = WeightDictMerkleTree.from_state_dict(self.dummy_dict)

        diff_weights, diff_nodes = tree1.diff(tree2)
        self.assertEqual(diff_weights, set())
        self.assertEqual(diff_nodes, {THIS: set(), OTHER: set()})

    def test_diff_same_architecture_diff_weights(self):
        tree1 = WeightDictMerkleTree.from_state_dict(self.dummy_dict)
        tree2 = WeightDictMerkleTree.from_state_dict(self.dummy_dict2)

        diff_weights, diff_nodes = tree1.diff(tree2)
        self.assertEqual(diff_weights, {LAYER_4, LAYER_5})
        self.assertEqual(diff_nodes, {THIS: set(), OTHER: set()})

    def test_added_layers(self):
        tree1 = WeightDictMerkleTree.from_state_dict(self.dummy_dict)
        tree2 = WeightDictMerkleTree.from_state_dict(self.dummy_dict3)

        diff_weights, diff_nodes = tree1.diff(tree2)
        self.assertEqual(diff_weights, set())
        self.assertEqual(diff_nodes[THIS], set())
        self.assertEqual(diff_nodes[OTHER], {LAYER_6, LAYER_7})

    def test_added_layers_and_diff_layers(self):
        tree1 = WeightDictMerkleTree.from_state_dict(self.dummy_dict)
        tree2 = WeightDictMerkleTree.from_state_dict(self.dummy_dict4)

        diff_weights, diff_nodes = tree1.diff(tree2)
        self.assertEqual(diff_weights, {LAYER_1})
        self.assertEqual(diff_nodes[THIS], set())
        self.assertEqual(diff_nodes[OTHER], {LAYER_6, LAYER_7})

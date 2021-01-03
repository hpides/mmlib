import unittest

import torch
from torch import nn
from torchvision import models

from mmlib.deterministic import set_deterministic
from mmlib.model_equals import imagenet_input
from mmlib.probe import ProbeSummary, probe_inference, imagenet_target, probe_training


class TestProbe(unittest.TestCase):
    def test_equal_empty_summaries(self):
        s1 = ProbeSummary()
        s2 = ProbeSummary()

        self.assertEqual(s1, s2)

    def test_equal_attributed_summaries(self):
        model1 = models.alexnet(pretrained=True)
        dummy_input = imagenet_input()
        summary1 = probe_inference(model1, dummy_input)
        summary2 = probe_inference(model1, dummy_input)

        self.assertEqual(summary1, summary2)

    def test_compare_order_summaries(self):
        model1 = models.alexnet(pretrained=True)
        dummy_input = imagenet_input()
        summary1 = probe_inference(model1, dummy_input)
        summary2 = probe_inference(model1, dummy_input)

        self.assertEqual(summary2, summary1)

    def test_unequal_summaries(self):
        dummy_input = imagenet_input()
        model1 = models.alexnet(pretrained=True)
        summary1 = probe_inference(model1, dummy_input)
        model2 = models.resnet18(pretrained=True)
        summary2 = probe_inference(model2, dummy_input)
        summary3 = ProbeSummary()

        self.assertNotEqual(summary1, summary2)
        self.assertNotEqual(summary1, summary3)
        self.assertNotEqual(summary2, summary3)

    def test_nondeterministic_computation(self):
        model1 = models.alexnet(pretrained=True)
        model2 = models.alexnet(pretrained=True)

        dummy_input = imagenet_input()
        dummy_target = imagenet_target(dummy_input)
        loss_func = nn.CrossEntropyLoss()

        # two optimizer objects because they have internal state, e.g. update their momentum
        optimizer1 = torch.optim.SGD(model1.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)
        optimizer2 = torch.optim.SGD(model1.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)

        summary1 = probe_training(model1, dummy_input, optimizer1, loss_func, dummy_target)
        summary2 = probe_training(model2, dummy_input, optimizer2, loss_func, dummy_target)

        self.assertNotEqual(summary1, summary2)

    def test_deterministic_computation(self):
        model1 = models.alexnet(pretrained=True)
        model2 = models.alexnet(pretrained=True)

        dummy_input = imagenet_input()
        dummy_target = imagenet_target(dummy_input)
        loss_func = nn.CrossEntropyLoss()

        # two optimizer objects because they have internal state, e.g. update their momentum
        optimizer1 = torch.optim.SGD(model1.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)
        optimizer2 = torch.optim.SGD(model2.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)

        set_deterministic()
        summary1 = probe_training(model1, dummy_input, optimizer1, loss_func, dummy_target)
        set_deterministic()
        summary2 = probe_training(model2, dummy_input, optimizer2, loss_func, dummy_target)

        self.assertEqual(summary1, summary2)

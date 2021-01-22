import unittest

import torch
from torchvision import models

from mmlib.deterministic import set_deterministic
from mmlib.helper import imagenet_input
from mmlib.model_equal import state_dict_equal, equal


class TestStateDictEqual(unittest.TestCase):

    def test_empty_dicts(self):
        d1 = {}
        d2 = {}

        self.assertTrue(state_dict_equal(d1, d2))

    def test_same_dicts(self):
        tensor = torch.rand(3, 300, 400)

        d1 = {'test': tensor}

        self.assertTrue(state_dict_equal(d1, d1))

    def test_equal_dicts(self):
        tensor = torch.rand(3, 300, 400)

        d1 = {'test': tensor}
        d2 = {'test': tensor}

        self.assertTrue(state_dict_equal(d1, d2))

    def test_different_keys(self):
        tensor = torch.rand(3, 300, 400)

        d1 = {'test1': tensor}
        d2 = {'test2': tensor}

        self.assertFalse(state_dict_equal(d1, d2))

    def test_different_tensor(self):
        tensor1 = torch.rand(3, 300, 400)
        tensor2 = torch.rand(3, 300, 400)

        d1 = {'test': tensor1}
        d2 = {'test': tensor2}

        self.assertFalse(state_dict_equal(d1, d2))


class TestModelEqual(unittest.TestCase):

    def test_resnet18_pretrained(self):
        mod1 = models.resnet18(pretrained=True)
        mod2 = models.resnet18(pretrained=True)

        self.assertTrue(equal(mod1, mod2, imagenet_input))

    def test_resnet50_pretrained(self):
        mod1 = models.resnet50(pretrained=True)
        mod2 = models.resnet50(pretrained=True)

        self.assertTrue(equal(mod1, mod2, imagenet_input))

    def test_resnet152_pretrained(self):
        mod1 = models.resnet152(pretrained=True)
        mod2 = models.resnet152(pretrained=True)

        self.assertTrue(equal(mod1, mod2, imagenet_input))

    def test_vgg19_pretrained(self):
        mod1 = models.vgg19(pretrained=True)
        mod2 = models.vgg19(pretrained=True)

        self.assertTrue(equal(mod1, mod2, imagenet_input))

    def test_alexnet_pretrained(self):
        mod1 = models.alexnet(pretrained=True)
        mod2 = models.alexnet(pretrained=True)

        self.assertTrue(equal(mod1, mod2, imagenet_input))

    def test_resnet18_resnet152_pretrained(self):
        mod1 = models.resnet18(pretrained=True)
        mod2 = models.resnet152(pretrained=True)

        self.assertFalse(equal(mod1, mod2, imagenet_input))

    def test_not_pretrained(self):
        mod1 = models.resnet18()
        mod2 = models.resnet18()

        # we expect this to be false since the weight initialization is random
        self.assertFalse(equal(mod1, mod2, imagenet_input))

    def test_resnet18_not_pretrained_deterministic(self):
        set_deterministic()
        mod1 = models.resnet18()

        set_deterministic()
        mod2 = models.resnet18()

        # we expect this to be true, the weights are randomly initialized,
        # but we set the seeds before weight initialization
        self.assertTrue(equal(mod1, mod2, imagenet_input))

    def test_resnet152_not_pretrained_deterministic(self):
        set_deterministic()
        mod1 = models.resnet152()

        set_deterministic()
        mod2 = models.resnet152()

        # we expect this to be true, the weights are randomly initialized,
        # but we set the seeds before weight initialization
        self.assertTrue(equal(mod1, mod2, imagenet_input))

    def test_vgg19_not_pretrained_deterministic(self):
        set_deterministic()
        mod1 = models.vgg19()

        set_deterministic()
        mod2 = models.vgg19()

        # we expect this to be true, the weights are randomly initialized,
        # but we set the seeds before weight initialization
        self.assertTrue(equal(mod1, mod2, imagenet_input))

    def test_alexnet_not_pretrained_deterministic(self):
        set_deterministic()
        mod1 = models.alexnet()

        set_deterministic()
        mod2 = models.alexnet()

        # we expect this to be true, the weights are randomly initialized,
        # but we set the seeds before weight initialization
        self.assertTrue(equal(mod1, mod2, imagenet_input))

    def test_not_pretrained_deterministic_multiple_models(self):
        set_deterministic()
        alex1 = models.alexnet()
        resnet1 = models.resnet18()

        set_deterministic()
        alex2 = models.alexnet()
        resnet2 = models.resnet18()

        self.assertTrue(equal(alex1, alex2, imagenet_input))
        self.assertTrue(equal(resnet1, resnet2, imagenet_input))

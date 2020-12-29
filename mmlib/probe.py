from enum import Enum

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models

from collections import OrderedDict
import numpy as np

# The following code is inspired by https://github.com/sksq96/pytorch-summary
from mmlib.model_equals import imagenet_input


class probe_info(Enum):
    LAYER = 'layer'
    INPUT_SHAPE = 'input_shape'
    INPUT_HASH = 'input_hash'
    OUTPUT_SHAPE = 'output_shape'
    OUTPUT_HASH = 'output_hash'


def _should_register(model, module):
    return not isinstance(module, nn.Sequential) \
           and not isinstance(module, nn.ModuleList) \
           and not (module == model)


def _module_key(module, summary):
    class_name = str(module.__class__).split(".")[-1].split("'")[0]
    module_idx = len(summary)
    return "%s-%i" % (class_name, module_idx + 1)


def probe_reproducibility(model, input, output_info, device="cuda", forward=True, backward=False):
    def register_forward_hook(module, ):

        def hook(module, input, output):
            module_key = _module_key(module, summary)

            summary[module_key] = OrderedDict()

            summary[module_key][probe_info.INPUT_SHAPE.value] = list(input[0].shape)
            summary[module_key][probe_info.INPUT_HASH.value] = hash(str(input))
            summary[module_key][probe_info.OUTPUT_SHAPE.value] = list(output.shape)
            summary[module_key][probe_info.OUTPUT_HASH.value] = hash(str(output))

        if _should_register(model, module):
            hooks.append(module.register_forward_hook(hook))

    # TODO clean up code
    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_forward_hook)

    # make a forward pass
    model(input)

    # remove these hooks
    for h in hooks:
        h.remove()

    format_string = " ".join(["{:>20}"] * len(output_info))
    _print_header(output_info, format_string)
    for layer in summary:
        _print_layer(layer, summary, output_info, format_string)


def _print_layer(layer, summary, output_info, format_string):
    values = []
    output_info = output_info.copy()

    if probe_info.LAYER in output_info:
        output_info.remove(probe_info.LAYER)
        values.append(layer)

    values += [str(summary[layer][x.value]) for x in output_info]
    line = format_string.format(*values)
    print(line)


def _print_header(output_info, format_string):
    names = [x.value for x in output_info]

    print("-----------------------------------------------------------------------------------------------------------")
    print(format_string.format(*names))
    print("===========================================================================================================")


# TODO delete main
if __name__ == '__main__':
    # models = [models.alexnet, models.vgg19, models.resnet18, models.resnet50, models.resnet152]
    models = [models.resnet18]
    tensor1 = imagenet_input()

    for mod in models:
        model = mod()
        print('Model: {}'.format(mod.__name__))
        probe_reproducibility(
            model,
            tensor1,
            [probe_info.LAYER, probe_info.INPUT_SHAPE,probe_info.INPUT_HASH, probe_info.OUTPUT_SHAPE, probe_info.OUTPUT_HASH])
        print('\n\n')

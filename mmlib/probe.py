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

    return summary


def _print_layer(layer, summary, output_info):
    values = []
    output_info = output_info.copy()

    if probe_info.LAYER in output_info:
        output_info.remove(probe_info.LAYER)
        values.append(layer)

    values += [str(summary[layer][x.value]) for x in output_info]
    format_string = " ".join(["{:>20}"] * len(values))
    line = format_string.format(*values)
    print(line)

def _print_compare_layer(fields, layer, summary1, summary2):
    pass


def _print_header(header_fields):
    print("-----------------------------------------------------------------------------------------------------------")
    header_format_string = " ".join(["{:>20}"] * len(header_fields))
    print(header_format_string.format(*header_fields))
    print("===========================================================================================================")


def print_summary(summary, output_info):
    header_fields = [x.value for x in output_info]
    _print_header(header_fields)
    for layer in summary:
        _print_layer(layer, summary, output_info)





def compare_summaries(summary1, summary2, compare, common=None):
    assert summary1.keys() == summary2.keys(), 'summary keys dont match'
    assert len(compare) > 0, 'you have to compare at least one attribute'

    if not common:
        common = [probe_info.LAYER]
    if probe_info.LAYER not in common:
        common.insert(0, probe_info.LAYER)

    header_fields = []
    for com in common:
        header_fields.append(com.value)
    for comp in compare:
        header_fields.append(comp.value + '-1')
        header_fields.append(comp.value + '-2')

    _print_header(header_fields)

    fields = common + compare
    for layer in summary1:
        _print_compare_layer(fields, layer, summary1, summary2)



# TODO delete main
if __name__ == '__main__':
    # models = [models.alexnet, models.vgg19, models.resnet18, models.resnet50, models.resnet152]
    models = [models.resnet18]
    tensor1 = imagenet_input()

    for mod in models:
        model = mod()
        print('Model: {}'.format(mod.__name__))
        output_info = [probe_info.LAYER, probe_info.INPUT_SHAPE, probe_info.INPUT_HASH, probe_info.OUTPUT_SHAPE,
                       probe_info.OUTPUT_HASH]
        summary = probe_reproducibility(model, tensor1, output_info)
        print_summary(summary, output_info)
        # compare_summaries(summary, summary, [probe_info.INPUT_HASH])
        print('\n\n')

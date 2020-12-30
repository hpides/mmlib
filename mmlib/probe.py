from collections import OrderedDict
from enum import Enum

import torch
import torch.nn as nn
from colorama import Fore, Style
from torchvision import models

from mmlib.model_equals import imagenet_input

# The following code is inspired by https://github.com/sksq96/pytorch-summary

PLACE_HOLDER = "{:>20}"


class ProbeInfo(Enum):
    LAYER = 'layer'
    INPUT_SHAPE = 'input_shape'
    INPUT_HASH = 'input_hash'
    OUTPUT_SHAPE = 'output_shape'
    OUTPUT_HASH = 'output_hash'


def probe_reproducibility(model, input, device="cuda", forward=True, backward=False):
    def register_forward_hook(module, ):

        def hook(module, input, output):
            module_key = _module_key(module, summary)

            summary[module_key] = OrderedDict()

            summary[module_key][ProbeInfo.INPUT_SHAPE.value] = list(input[0].shape)
            summary[module_key][ProbeInfo.INPUT_HASH.value] = hash(str(input))
            summary[module_key][ProbeInfo.OUTPUT_SHAPE.value] = list(output.shape)
            summary[module_key][ProbeInfo.OUTPUT_HASH.value] = hash(str(output))

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


def print_summary(summary, output_info):
    header_fields = [x.value for x in output_info]
    _print_header(header_fields)
    for layer in summary:
        _print_layer(layer, summary, output_info)


def compare_summaries(summary1, summary2, compare, common=None):
    assert summary1.keys() == summary2.keys(), 'summary keys do not match'
    assert len(compare) > 0, 'you have to compare at least one attribute'

    if not common:
        common = [ProbeInfo.LAYER]
    if ProbeInfo.LAYER not in common:
        common.insert(0, ProbeInfo.LAYER)

    _print_compare_header(common, compare)

    fields = common + compare
    for layer in summary1:
        _print_compare_layer(fields, layer, summary1, summary2)


def _should_register(model, module):
    return not isinstance(module, nn.Sequential) \
           and not isinstance(module, nn.ModuleList) \
           and not (module == model)


def _module_key(module, summary):
    class_name = str(module.__class__).split(".")[-1].split("'")[0]
    module_idx = len(summary)
    return "%s-%i" % (class_name, module_idx + 1)


def _print_compare_header(common, compare):
    header_fields = []
    for com in common:
        header_fields.append(com.value)
    for comp in compare:
        header_fields.append(comp.value + '-1')
        header_fields.append(comp.value + '-2')
    _print_header(header_fields)


def _print_compare_layer(fields, layer, summary1, summary2):
    values = []
    fields = fields.copy()

    if ProbeInfo.LAYER in fields:
        fields.remove(ProbeInfo.LAYER)
        values.append(layer)

    for field in fields:
        v1 = summary1[layer][field.value]
        v2 = summary2[layer][field.value]

        color = Fore.GREEN
        if v1 != v2:
            color = Fore.RED

        values.append(v1)
        values.append(v2)

    format_string = " ".join([PLACE_HOLDER] * len(values))
    line = format_string.format(*values)
    print(color + line + Style.RESET_ALL)


def _print_layer(layer, summary, output_info):
    values = []
    output_info = output_info.copy()

    if ProbeInfo.LAYER in output_info:
        output_info.remove(ProbeInfo.LAYER)
        values.append(layer)

    values += [str(summary[layer][x.value]) for x in output_info]
    format_string = " ".join([PLACE_HOLDER] * len(values))
    line = format_string.format(*values)
    print(line)


def _print_header(header_fields):
    print("-----------------------------------------------------------------------------------------------------------")
    header_format_string = " ".join([PLACE_HOLDER] * len(header_fields))
    print(header_format_string.format(*header_fields))
    print("===========================================================================================================")


# TODO delete main
if __name__ == '__main__':
    # models = [models.alexnet, models.vgg19, models.resnet18, models.resnet50, models.resnet152]
    models = [models.resnet18]
    tensor1 = imagenet_input()

    for mod in models:
        model1 = mod()
        model2 = mod()
        print('Model: {}'.format(mod.__name__))
        output_info = [ProbeInfo.LAYER, ProbeInfo.INPUT_SHAPE, ProbeInfo.INPUT_HASH, ProbeInfo.OUTPUT_SHAPE,
                       ProbeInfo.OUTPUT_HASH]
        summary1 = probe_reproducibility(model1, tensor1)
        summary2 = probe_reproducibility(model2, tensor1)
        # print_summary(summary, output_info)
        compare_summaries(summary1, summary2, [ProbeInfo.INPUT_HASH, ProbeInfo.OUTPUT_HASH])
        print('\n\n')

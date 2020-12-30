from collections import OrderedDict
from enum import Enum

import torch
import torch.nn as nn
from colorama import Fore, Style
from torchvision import models

from mmlib.model_equals import imagenet_input

# The following code is inspired by https://github.com/sksq96/pytorch-summary

PLACE_HOLDER_LEN = 20
PLACE_HOLDER = "{:>" + str(PLACE_HOLDER_LEN) + "}"


class ProbeInfo(Enum):
    LAYER = 'layer'
    INPUT_SHAPE = 'input_shape'
    INPUT_HASH = 'input_hash'
    OUTPUT_SHAPE = 'output_shape'
    OUTPUT_HASH = 'output_hash'


class ProbeMode(Enum):
    INFERENCE = 1
    TRAINING = 2


def probe_inference(model, inp, device="cuda"):
    return probe_reproducibility(model, inp, ProbeMode.INFERENCE, device=device)


def probe_training(model, inp, optimizer, loss_func, target, device="cuda"):
    return probe_reproducibility(model, inp, ProbeMode.TRAINING, optimizer=optimizer, loss_func=loss_func,
                                 target=target, device=device)


def probe_reproducibility(model, inp, mode, optimizer=None, loss_func=None, target=None, device="cuda"):
    if mode == ProbeMode.TRAINING:
        assert optimizer is not None, 'for training mode a optimizer is needed'
        assert loss_func is not None, 'for training mode a loss_func is needed'
        assert target is not None, 'for training mode a target is needed'

    def register_forward_hook(module, ):

        def hook(module, input, output):
            module_key = _module_key(module, summary)

            summary[module_key] = OrderedDict()

            summary[module_key][ProbeInfo.INPUT_SHAPE.value] = str(list(input[0].shape))
            summary[module_key][ProbeInfo.INPUT_HASH.value] = str(hash(str(input)))
            summary[module_key][ProbeInfo.OUTPUT_SHAPE.value] = str(list(output.shape))
            summary[module_key][ProbeInfo.OUTPUT_HASH.value] = str(hash(str(output)))

        if _should_register(model, module):
            hooks.append(module.register_forward_hook(hook))

    dtype = _dtype(device)

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_forward_hook)

    if mode == ProbeMode.INFERENCE:
        model.eval()
        model(inp)
    elif mode == ProbeMode.TRAINING:
        model.train()
        output = model(inp)

        # just set dummpy loss and optimizer
        loss = loss_func(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

    # make sure Layer info is included
    if not common:
        common = [ProbeInfo.LAYER]
    if ProbeInfo.LAYER not in common:
        common.insert(0, ProbeInfo.LAYER)

    _print_compare_header(common, compare)

    for layer in summary1:
        _print_compare_layer(common, compare, layer, summary1, summary2)


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


def _print_compare_layer(common, compare, layer, summary1, summary2):
    common = common.copy()
    line = ""

    if ProbeInfo.LAYER in common:
        common.remove(ProbeInfo.LAYER)
        line += PLACE_HOLDER.format(layer) + " "

    for field in common:
        value_ = summary1[layer][field.value]
        line += PLACE_HOLDER.format(value_) + " "

    for field in compare:
        v1 = summary1[layer][field.value]
        v2 = summary2[layer][field.value]

        color = Fore.GREEN
        if v1 != v2:
            color = Fore.RED

        line += color + " ".join([PLACE_HOLDER] * 2).format(v1, v2) + Style.RESET_ALL + " "

    print(line)


def _print_layer(layer, summary, output_info):
    values = []
    output_info = output_info.copy()

    if ProbeInfo.LAYER in output_info:
        output_info.remove(ProbeInfo.LAYER)
        values.append(layer)

    values += [summary[layer][x.value] for x in output_info]
    format_string = " ".join([PLACE_HOLDER] * len(values))
    line = format_string.format(*values)
    print(line)


def _print_header(header_fields):
    format_string = "=".join([PLACE_HOLDER] * len(header_fields))
    insert = ["=" * PLACE_HOLDER_LEN] * len(header_fields)
    devider = format_string.format(*insert)

    print(devider)
    header_format_string = " ".join([PLACE_HOLDER] * len(header_fields))
    print(header_format_string.format(*header_fields))
    print(devider)


def _dtype(device):
    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"
    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    return dtype


# TODO delete main
if __name__ == '__main__':
    models = [models.alexnet, models.vgg19, models.resnet18, models.resnet50, models.resnet152]
    # models = [models.resnet18]
    tensor1 = imagenet_input()

    for mod in models:
        model1 = mod()
        model2 = model1
        print('Model: {}'.format(mod.__name__))
        output_info = [ProbeInfo.LAYER, ProbeInfo.INPUT_SHAPE, ProbeInfo.INPUT_HASH, ProbeInfo.OUTPUT_SHAPE,
                       ProbeInfo.OUTPUT_HASH]
        # summary1 = probe_inference(model1, tensor1)
        # summary2 = probe_inference(model2, tensor1)
        # print_summary(summary1, output_info)
        # compare_summaries(summary1, summary2, [ProbeInfo.INPUT_HASH, ProbeInfo.OUTPUT_HASH],
        #                   common=[ProbeInfo.LAYER, ProbeInfo.INPUT_SHAPE])
        print('\n\n')

        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model1.parameters(), 1e-4,
                                    momentum=0.9,
                                    weight_decay=1e-4)
        dummy_target = torch.tensor([1])

        model1 = mod(pretrained=True)
        model2 = mod(pretrained=True)
        sum_train1 = probe_training(model1, inp=tensor1, optimizer=optimizer, loss_func=loss_func, target=dummy_target)
        sum_train2 = probe_training(model2, inp=tensor1, optimizer=optimizer, loss_func=loss_func, target=dummy_target)
        compare_summaries(sum_train1, sum_train2, [ProbeInfo.INPUT_HASH, ProbeInfo.OUTPUT_HASH],
                          common=[ProbeInfo.LAYER, ProbeInfo.INPUT_SHAPE])

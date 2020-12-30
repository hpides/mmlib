from collections import OrderedDict
from enum import Enum

import torch.nn as nn
from colorama import Fore, Style

PLACE_HOLDER_LEN = 22
PLACE_HOLDER = "{:>" + str(PLACE_HOLDER_LEN) + "}"


class ProbeInfo(Enum):
    LAYER = 'layer'

    INPUT_SHAPE = 'input_shape'
    INPUT_HASH = 'input_hash'
    OUTPUT_SHAPE = 'output_shape'
    OUTPUT_HASH = 'output_hash'

    GRAD_INPUT_SHAPE = 'grad_input_shape'
    GRAD_INPUT_HASH = 'grad_input_hash'
    GRAD_OUTPUT_SHAPE = 'grad_output_shape'
    GRAD_OUTPUT_HASH = 'grad_output_hash'


class ProbeMode(Enum):
    INFERENCE = 1
    TRAINING = 2


def probe_inference(model, inp, device="cuda"):
    return probe_reproducibility(model, inp, ProbeMode.INFERENCE)


def probe_training(model, inp, optimizer, loss_func, target):
    return probe_reproducibility(model, inp, ProbeMode.TRAINING, optimizer=optimizer, loss_func=loss_func,
                                 target=target)


def probe_reproducibility(model, inp, mode, optimizer=None, loss_func=None, target=None):
    if mode == ProbeMode.TRAINING:
        assert optimizer is not None, 'for training mode a optimizer is needed'
        assert loss_func is not None, 'for training mode a loss_func is needed'
        assert target is not None, 'for training mode a target is needed'

    # The following code is inspired by https://github.com/sksq96/pytorch-summary
    def register_forward_hook(module, ):

        def hook(module, input, output):
            module_key = _module_key(module, forward_layer_keys)
            forward_layer_keys.append(module_key)

            summary[module_key] = OrderedDict()

            summary[module_key][ProbeInfo.INPUT_SHAPE.value] = str(list(input[0].shape))
            summary[module_key][ProbeInfo.INPUT_HASH.value] = str(hash(str(input)))
            summary[module_key][ProbeInfo.OUTPUT_SHAPE.value] = str(list(output.shape))
            summary[module_key][ProbeInfo.OUTPUT_HASH.value] = str(hash(str(output)))

        if _should_register(model, module):
            hooks.append(module.register_forward_hook(hook))

    def register_backward_hook(module, ):

        def hook(module, grad_input, grad_output):
            module_key = _module_key(module, backward_layer_keys)
            backward_layer_keys.append(module_key)

            summary[module_key][ProbeInfo.GRAD_INPUT_SHAPE.value] = str(_shape_list(grad_input))
            summary[module_key][ProbeInfo.GRAD_INPUT_HASH.value] = str(hash(str(grad_input)))
            summary[module_key][ProbeInfo.GRAD_OUTPUT_SHAPE.value] = str(_shape_list(grad_output))
            summary[module_key][ProbeInfo.GRAD_OUTPUT_HASH.value] = str(hash(str(grad_output)))

        if _should_register(model, module):
            hooks.append(module.register_backward_hook(hook))

    # create properties
    summary = OrderedDict()
    forward_layer_keys = []
    backward_layer_keys = []
    hooks = []

    # register forward hook
    model.apply(register_forward_hook)
    model.apply(register_backward_hook)

    if mode == ProbeMode.INFERENCE:
        model.eval()
        model(inp)
    elif mode == ProbeMode.TRAINING:
        model.train()
        output = model(inp)
        loss = loss_func(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # remove these hooks
    for h in hooks:
        h.remove()

    return _replace_hash_by_count(summary)


def print_summary(summary, summary_info):
    header_fields = [x.value for x in summary_info]
    _print_header(header_fields)
    for layer in summary:
        _print_layer(layer, summary, summary_info)


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


def _module_key(module, layer_keys):
    class_name = str(module.__class__).split(".")[-1].split("'")[0]
    h = hash(module)
    return "{}-{}".format(class_name, hash(module))


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
        if field.value in summary1[layer] and field.value in summary2[layer]:
            v1 = summary1[layer][field.value]
            v2 = summary2[layer][field.value]
        else:
            v1 = v2 = '---'

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


def _replace_hash_by_count(ordered_dict):
    result = OrderedDict()

    for i, k in enumerate(ordered_dict):
        new_key = "{}-{}".format(k.split('-')[0], str(i + 1))
        result[new_key] = ordered_dict[k]

    return result


def _shape_list(tensor_tuple):
    result = []
    for t in tensor_tuple:
        if t is None:
            result.append([])
        else:
            result.append(t.shape)

    return result

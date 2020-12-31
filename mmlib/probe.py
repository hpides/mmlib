from enum import Enum

import torch
import torch.nn as nn


class ProbeInfo(Enum):
    LAYER_ID = 'layer_id'

    FORWARD_INDEX = 'forward_index'
    BACKWARD_INDEX = 'backward_index'

    LAYER_NAME = 'layer_name'

    INPUT_SHAPE = 'input_shape'
    INPUT_TENSOR = 'input_hash'
    OUTPUT_SHAPE = 'output_shape'
    OUTPUT_TENSOR = 'output_hash'

    GRAD_INPUT_SHAPE = 'grad_input_shape'
    GRAD_INPUT_TENSOR = 'grad_input_hash'
    GRAD_OUTPUT_SHAPE = 'grad_output_shape'
    GRAD_OUTPUT_TENSOR = 'grad_output_hash'


class ProbeMode(Enum):
    INFERENCE = 1
    TRAINING = 2


class ProbeSummary:
    PLACE_HOLDER_LEN = 22
    PLACE_HOLDER = "{:>" + str(PLACE_HOLDER_LEN) + "}"

    def __init__(self):
        self.summary = {}

    def add_attribute(self, module_key: str, attribute: ProbeInfo, value):
        if module_key not in self.summary:
            self.summary[module_key] = {}

        self.summary[module_key][attribute] = value

    def print_summary(self, info: [ProbeInfo]):
        self._print_header(info)
        for layer_key, layer_info in self.summary.items():
            self._print_summary_layer(layer_info, info)

    # def print_sum(self):
    #     # TODO move
    #     if True:
    #         torch.set_printoptions(profile='full')
    #     else:
    #         torch.set_printoptions(profile='default')
    #     pass

    def _print_header(self, info):
        header_fields = [x.value for x in info]

        format_string = "=".join([self.PLACE_HOLDER] * len(header_fields))
        insert = ["=" * self.PLACE_HOLDER_LEN] * len(header_fields)
        devider = format_string.format(*insert)

        print(devider)
        header_format_string = " ".join([self.PLACE_HOLDER] * len(header_fields))
        print(header_format_string.format(*header_fields))
        print(devider)

    def _print_summary_layer(self, layer_info, info):
        values = []
        for i in info:
            lay_inf = layer_info[i]
            if torch.is_tensor(lay_inf) or isinstance(lay_inf, tuple) and torch.is_tensor(lay_inf[0]):
                # TODO fix hashing here
                values.append(str(hash(str(lay_inf))))
            else:
                values.append(str(lay_inf))

        format_string = " ".join([self.PLACE_HOLDER] * len(values))
        line = format_string.format(*values)
        print(line)


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

    def register_forward_hook(module, ):

        def hook(module, input, output):
            layer_name = _layer_name(module)
            layer_key = _layer_key(layer_name, module)

            forward_layer_keys.append(layer_key)

            summary.add_attribute(layer_key, ProbeInfo.FORWARD_INDEX, len(forward_layer_keys))
            summary.add_attribute(layer_key, ProbeInfo.LAYER_NAME, layer_name)
            summary.add_attribute(layer_key, ProbeInfo.INPUT_SHAPE, _shape_list(input))
            summary.add_attribute(layer_key, ProbeInfo.INPUT_TENSOR, input)
            summary.add_attribute(layer_key, ProbeInfo.OUTPUT_SHAPE, _shape_list(output))
            summary.add_attribute(layer_key, ProbeInfo.OUTPUT_TENSOR, output)

        if _should_register(model, module):
            hooks.append(module.register_forward_hook(hook))

    def register_backward_hook(module, ):

        def hook(module, grad_input, grad_output):
            layer_name = _layer_name(module)
            layer_key = _layer_key(layer_name, module)

            backward_layer_keys.append(layer_key)

            summary.add_attribute(layer_key, ProbeInfo.BACKWARD_INDEX, len(backward_layer_keys))
            summary.add_attribute(layer_key, ProbeInfo.LAYER_NAME, layer_name)
            summary.add_attribute(layer_key, ProbeInfo.GRAD_INPUT_SHAPE, _shape_list(grad_input))
            summary.add_attribute(layer_key, ProbeInfo.GRAD_INPUT_TENSOR, grad_input)
            summary.add_attribute(layer_key, ProbeInfo.GRAD_OUTPUT_SHAPE, _shape_list(grad_output))
            summary.add_attribute(layer_key, ProbeInfo.GRAD_OUTPUT_TENSOR, grad_output)

        if _should_register(model, module):
            hooks.append(module.register_backward_hook(hook))

    # create properties
    summary = ProbeSummary()
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

    return summary


# def compare_summaries(summary1, summary2, compare, common=None):
#     assert summary1.keys() == summary2.keys(), 'summary keys do not match'
#     assert len(compare) > 0, 'you have to compare at least one attribute'
#
#     # make sure Layer info is included
#     if not common:
#         common = [ProbeInfo.LAYER]
#     if ProbeInfo.LAYER not in common:
#         common.insert(0, ProbeInfo.LAYER)
#
#     _print_compare_header(common, compare)
#
#     for layer in summary1:
#         _print_compare_layer(common, compare, layer, summary1, summary2)


def _should_register(model, module):
    return not isinstance(module, nn.Sequential) \
           and not isinstance(module, nn.ModuleList) \
           and not (module == model)


# def _print_compare_header(common, compare):
#     header_fields = []
#     for com in common:
#         header_fields.append(com.value)
#     for comp in compare:
#         header_fields.append(comp.value + '-1')
#         header_fields.append(comp.value + '-2')
#     _print_header(header_fields)
#
#
# def _print_compare_layer(common, compare, layer, summary1, summary2):
#     common = common.copy()
#     line = ""
#
#     if ProbeInfo.LAYER in common:
#         common.remove(ProbeInfo.LAYER)
#         line += PLACE_HOLDER.format(layer) + " "
#
#     for field in common:
#         value_ = summary1[layer][field.value]
#         line += PLACE_HOLDER.format(value_) + " "
#
#     for field in compare:
#         if field.value in summary1[layer] and field.value in summary2[layer]:
#             v1 = summary1[layer][field.value]
#             v2 = summary2[layer][field.value]
#         else:
#             v1 = v2 = '---'
#
#         color = Fore.GREEN
#         if v1 != v2:
#             color = Fore.RED
#
#         line += color + " ".join([PLACE_HOLDER] * 2).format(v1, v2) + Style.RESET_ALL + " "
#
#     print(line)


def _layer_name(module):
    return str(module.__class__).split(".")[-1].split("'")[0]


def _layer_key(layer_name, module):
    return "{}-{}".format(layer_name, hash(module))


def _shape_list(tensor_tuple):
    result = []
    for t in tensor_tuple:
        if t is None:
            result.append([])
        else:
            result.append(list(t.shape))

    return result

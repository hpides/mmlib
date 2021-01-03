import random
from enum import Enum

import torch
import torch.nn as nn
from colorama import Fore, Style


class ProbeInfo(Enum):
    LAYER_ID = 'layer_id'

    FORWARD_INDEX = 'forward_index'
    BACKWARD_INDEX = 'backward_index'

    LAYER_NAME = 'layer_name'

    INPUT_SHAPE = 'input_shape'
    INPUT_TENSOR = 'input_tensor'
    OUTPUT_SHAPE = 'output_shape'
    OUTPUT_TENSOR = 'output_tensor'

    GRAD_INPUT_SHAPE = 'grad_input_shape'
    GRAD_INPUT_TENSOR = 'grad_input_tensor'
    GRAD_OUTPUT_SHAPE = 'grad_output_shape'
    GRAD_OUTPUT_TENSOR = 'grad_output_tensor'


class ProbeMode(Enum):
    INFERENCE = 1
    TRAINING = 2


class ProbeSummary:
    """
    Object of this class represent the results of a probing run
    """
    PLACE_HOLDER_LEN = 25
    PLACE_HOLDER = "{:>" + str(PLACE_HOLDER_LEN) + "}"
    DIFF = 'diff'
    SAME = 'same'

    def __init__(self, summary_path=None):
        if summary_path:
            self.load(summary_path)
        else:
            self.summary = {}

    def __eq__(self, other):
        try:
            for layer_key, layer_info in self.summary.items():
                other_info = self._find_forward_index(layer_info[ProbeInfo.FORWARD_INDEX], other.summary)
                for info_key, info_value in layer_info.items():
                    other_info_value = other_info[info_key]
                    if not self._compare_values(info_value, other_info_value):
                        return False
            return True

        except TypeError:
            # If there is a Type Error most likely the summaries do not have the same fields and are thus not equal.
            # Even if this is not the case if there is a TypeError they can't be equal.
            return False

    def add_attribute(self, module_key: str, attribute: ProbeInfo, value):
        """
        Adds an attribute the the summary.
        :param module_key: The key identifying the module that produced the input.
        :param attribute: The attribute that should be stored.
        :param value: The value of the given attribute that should be stored.
        """
        if module_key not in self.summary:
            self.summary[module_key] = {}

        self.summary[module_key][attribute] = value

    def print_summary(self, info: [ProbeInfo]):
        """
        Prints the summary object.
        :param info: The fields that should be included in the summary, given as a list of ProbeInfo Enums.
        """
        self._print_hashwarning(info)
        self._print_header([x.value for x in info])
        for layer_key, layer_info in self.summary.items():
            self._print_summary_layer(layer_info, info)

    def compare_to(self, other_summary, common: [ProbeInfo], compare: [ProbeInfo]):
        """
        Compares a given ProbeSummary object to the self object and prints color coded per layer comparison.
        The parameters common and compare can be used to customize the output but do not influence the first line saying
        if the summaries differ or are the same.
        :param other_summary: The summary to compare to.
        :param common: The fields that should be printed for readability but shouldn't be compared.
        :param compare: The fields that should be compared.
        """
        output = 'Other summary is: '
        if self == other_summary:
            output += Fore.GREEN + self.SAME + Style.RESET_ALL
        else:
            output += Fore.RED + self.DIFF + Style.RESET_ALL
        print(output)

        self._print_compare_header(common, compare)
        for layer_key, layer_info in self.summary.items():
            self._print_compare_layer(common, compare, layer_info, other_summary)

    def save(self, path):
        """
        Saves an object to a disk file.
        :param path: The path to store to.
        """
        torch.save(self.summary, path)

    def load(self, path):
        """
        Loads an object saved with :func:`mmlib.probe.save` from a file.
        :param path: The path to load from.
        """
        self.summary = torch.load(path)

    def _print_header(self, fields):
        format_string = "=".join([self.PLACE_HOLDER] * len(fields))
        insert = ["=" * self.PLACE_HOLDER_LEN] * len(fields)
        divider = format_string.format(*insert)

        print(divider)
        header_format_string = " ".join([self.PLACE_HOLDER] * len(fields))
        print(header_format_string.format(*fields))
        print(divider)

    def _print_summary_layer(self, layer_info, info):
        values = []
        for i in info:
            values.append(self._layer_info_str(layer_info[i]))

        self._print_layer(values)

    def _print_layer(self, values):
        format_string = " ".join([self.PLACE_HOLDER] * len(values))
        line = format_string.format(*values)
        print(line)

    def _layer_info_str(self, layer_info):
        if self._tensor_or_tensor_tuple(layer_info) or len(str(layer_info)) > self.PLACE_HOLDER_LEN:
            # if the representation of the info is to long, print a has value
            return str(hash(str(layer_info)))
        else:
            return str(layer_info)

    def _tensor_or_tensor_tuple(self, value):
        return torch.is_tensor(value) or isinstance(value, tuple) and torch.is_tensor(value[0])

    def _print_compare_layer(self, common, compare, layer_info, other_summary):
        layer_info = layer_info
        other_layer_info = self._find_forward_index(layer_info[ProbeInfo.FORWARD_INDEX], other_summary.summary)

        line = ''
        for com in common:
            value_ = self._layer_info_str(layer_info[com])
            line += self.PLACE_HOLDER.format(value_) + " "

        for comp in compare:
            v1 = layer_info[comp]
            v2 = other_layer_info[comp]

            message = self.SAME
            color = Fore.GREEN
            if not self._compare_values(v1, v2):
                color = Fore.RED
                message = self.DIFF

            line += color + self.PLACE_HOLDER.format(message) + Style.RESET_ALL + " "

        print(line)

    def _print_compare_header(self, common, compare):
        header_fields = []
        for com in common:
            header_fields.append(com.value)
        for comp in compare:
            header_fields.append(comp.value + '-comp')
        self._print_header(header_fields)

    def _find_forward_index(self, index, other_summary):
        for _, info in other_summary.items():
            if info[ProbeInfo.FORWARD_INDEX] == index:
                return info

    def _compare_values(self, v1, v2):
        if isinstance(v1, tuple) and isinstance(v2, tuple):
            result = True
            for i in range(len(v1)):
                result = result and self._compare_values(v1[i], v2[i])
            return result
        elif torch.is_tensor(v1) and torch.is_tensor(v2):
            return torch.equal(v1, v2)
        else:
            return v1 == v2

    def _print_hashwarning(self, fields: [ProbeInfo]):
        if any('shape' in x.value or 'tensor' in x.value for x in fields):
            print(Fore.YELLOW, 'Warning: Same hashes don\'t have to mean that values are exactly the same. They '
                               'should be seen as an indicator.', Style.RESET_ALL)


def probe_inference(model, inp):
    """
    Probes the inference of a given model.
    :param model: The model to probe.
    :param inp: The model input to use.
    :return: A ProbeSummary object.
    """
    return _probe_reproducibility(model, inp, ProbeMode.INFERENCE)


def probe_training(model, inp, optimizer, loss_func, target):
    """
    Probes the training of a given model.
    :param model: The model to probe.
    :param inp: The model input to use.
    :param optimizer: The optimizer to use.
    :param loss_func: The loss function to use.
    :param target: The target data to use.
    :return: A ProbeSummary object.
    """
    return _probe_reproducibility(model, inp, ProbeMode.TRAINING, optimizer=optimizer, loss_func=loss_func,
                                  target=target)


def _probe_reproducibility(model, inp, mode, optimizer=None, loss_func=None, target=None):
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


def imagenet_target(dummy_input):
    batch_size = dummy_input.shape[0]
    batch = []
    for i in range(batch_size):
        batch.append(random.randint(1, 999))
    return torch.tensor(batch)


def _should_register(model, module):
    return not isinstance(module, nn.Sequential) \
           and not isinstance(module, nn.ModuleList) \
           and not (module == model)


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

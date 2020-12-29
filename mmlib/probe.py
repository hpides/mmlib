import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models


from collections import OrderedDict
import numpy as np

# The following code is inspired by https://github.com/sksq96/pytorch-summary
from mmlib.model_equals import imagenet_input


def _should_register(model, module):
    return not isinstance(module, nn.Sequential)\
           and not isinstance(module, nn.ModuleList)\
           and not (module == model)

def probe_reproducibility(model, input, device="cuda", forward=True, backward=False):

    def register_forward_hook(module, ):

        def hook(module, input, output):
            module_key = _module_key(module)

            summary[module_key] = OrderedDict()

            summary[module_key]["input_shape"] = list(input[0].shape)
            summary[module_key]["output_shape"] = list(output.shape)

        def _module_key(module):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            return "%s-%i" % (class_name, module_idx + 1)

        if _should_register(model, module):
            hooks.append(module.register_forward_hook(hook))

    # TODO define enum for device
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

    print("-----------------------------------------------------------------------------------------------------------")
    line_new = "{:>20} {:>20} {:>20}".format("Layer (type)", "Input Shape", "Output Shape", "Param #")
    print(line_new)
    print("===========================================================================================================")
    total_output = 0
    for layer in summary:
        # input_shape, output_shape, trainable
        line_new = "{:>20}  {:>25} {:>25}".format(
            layer,
            str(summary[layer]["input_shape"]),
            str(summary[layer]["output_shape"])
        )
        total_output += np.prod(summary[layer]["output_shape"])
        print(line_new)


# TODO delete main
if __name__ == '__main__':
    # models = [models.alexnet, models.vgg19, models.resnet18, models.resnet50, models.resnet152]
    models = [models.resnet18]
    tensor1 = imagenet_input()

    for mod in models:
        model = mod()
        print('Model: {}'.format(mod.__name__))
        probe_reproducibility(model, tensor1)
        print('\n\n')

import torch
from torch import nn
from torchvision import models

from mmlib.deterministic import set_deterministic
from mmlib.helper import imagenet_input, imagenet_target
from mmlib.model_equals import equals, whitebox_equals, blackbox_equals
from mmlib.probe import ProbeInfo, probe_inference, probe_training

MODEL = models.alexnet


def summary(device, forward_indices=None):
    model = MODEL(pretrained=True)
    summary_info = [ProbeInfo.LAYER_NAME, ProbeInfo.FORWARD_INDEX, ProbeInfo.INPUT_SHAPE, ProbeInfo.INPUT_TENSOR,
                    ProbeInfo.OUTPUT_SHAPE, ProbeInfo.OUTPUT_TENSOR]
    dummy_input = imagenet_input()

    # generate the summary using a dummy input
    summary = probe_inference(model, dummy_input, device, forward_indices=forward_indices)

    summary.print_summary(summary_info)


def forward_compare(device, forward_indices=None):
    model1 = MODEL(pretrained=True)
    model2 = MODEL(pretrained=True)

    dummy_input = imagenet_input()

    summary1 = probe_inference(model1, dummy_input, device, forward_indices=forward_indices)
    summary2 = probe_inference(model2, dummy_input, device, forward_indices=forward_indices)

    # fields that should for sure be the same
    common = [ProbeInfo.LAYER_NAME, ProbeInfo.FORWARD_INDEX]

    # fields where we might expect different values
    compare = [ProbeInfo.INPUT_SHAPE, ProbeInfo.INPUT_TENSOR, ProbeInfo.OUTPUT_TENSOR]

    # print the comparison of summary1 and summary2
    summary1.compare_to(summary2, common, compare)


def backward_compare(device, forward_indices=None):
    model1 = MODEL(pretrained=True)
    model2 = MODEL(pretrained=True)

    dummy_input = imagenet_input()
    dummy_target = imagenet_target(dummy_input)
    loss_func = nn.CrossEntropyLoss()

    # two optimizer objects because they have internal state, e.g. update their momentum
    optimizer1 = torch.optim.SGD(model1.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)
    optimizer2 = torch.optim.SGD(model2.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)

    summary1 = probe_training(model1, dummy_input, optimizer1, loss_func, dummy_target, device,
                              forward_indices=forward_indices)
    summary2 = probe_training(model2, dummy_input, optimizer2, loss_func, dummy_target, device,
                              forward_indices=forward_indices)

    # fields that should for sure be the same
    common = [ProbeInfo.LAYER_NAME]

    # fields where we might expect different values
    compare = [ProbeInfo.INPUT_TENSOR, ProbeInfo.OUTPUT_TENSOR, ProbeInfo.GRAD_INPUT_TENSOR,
               ProbeInfo.GRAD_OUTPUT_TENSOR]

    # print the comparison of summary1 and summary2
    summary1.compare_to(summary2, common, compare)


def deterministic_backward_compare(device, forward_indices=None):
    dummy_input = imagenet_input()
    dummy_target = imagenet_target(dummy_input)
    loss_func = nn.CrossEntropyLoss()

    set_deterministic()
    model1 = MODEL(pretrained=True)
    optimizer1 = torch.optim.SGD(model1.parameters(), 1e-3)
    summary1 = probe_training(model1, dummy_input, optimizer1, loss_func, dummy_target, device,
                              forward_indices=forward_indices)

    set_deterministic()
    model2 = MODEL(pretrained=True)
    optimizer2 = torch.optim.SGD(model2.parameters(), 1e-3)
    summary2 = probe_training(model2, dummy_input, optimizer2, loss_func, dummy_target, device,
                              forward_indices=forward_indices)

    # fields that should for sure be the same
    common = [ProbeInfo.LAYER_NAME]

    # fields where we might expect different values
    compare = [ProbeInfo.INPUT_TENSOR, ProbeInfo.OUTPUT_TENSOR, ProbeInfo.GRAD_INPUT_TENSOR,
               ProbeInfo.GRAD_OUTPUT_TENSOR]

    summary1.compare_to(summary2, common, compare)

    # also the models should be equal
    blackbox_equal = blackbox_equals(model1, model2, imagenet_input)
    whitebox_equal = whitebox_equals(model1, model2)
    models_are_equal = equals(model1, model2, imagenet_input)
    print()
    print('Also the models should be the same - compare the models')
    print('models_are_equal (blackbox): {}'.format(blackbox_equal))
    print('models_are_equal (whitebox): {}'.format(whitebox_equal))
    print('models_are_equal: {}'.format(models_are_equal))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device used: {}'.format(device))

    # show only the first 25 layers
    forward_indices = list(range(1, 25))

    summary(device, forward_indices)
    print('\n\n\n')
    forward_compare(device, forward_indices)
    print('\n\n\n')
    backward_compare(device, forward_indices)
    print('\n\n\n')
    deterministic_backward_compare(device, forward_indices)

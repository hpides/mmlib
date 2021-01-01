import torch
from torch import nn
from torchvision import models

from mmlib.deterministic import set_deterministic
from mmlib.model_equals import imagenet_input
from mmlib.probe import ProbeInfo, probe_inference, probe_training

MODEL = models.alexnet


def summary():
    model = MODEL(pretrained=True)
    summary_info = [ProbeInfo.LAYER_NAME, ProbeInfo.FORWARD_INDEX, ProbeInfo.INPUT_SHAPE, ProbeInfo.INPUT_TENSOR,
                    ProbeInfo.OUTPUT_SHAPE,
                    ProbeInfo.OUTPUT_TENSOR]
    dummy_input = imagenet_input()

    # generate the summary using a dummy input
    summary = probe_inference(model, dummy_input)

    summary.print_summary(summary_info)


def forward_compare():
    model1 = MODEL(pretrained=True)
    model2 = MODEL(pretrained=True)

    dummy_input = imagenet_input()

    summary1 = probe_inference(model1, dummy_input)
    summary2 = probe_inference(model2, dummy_input)

    # fields that should for sure be the same
    common = [ProbeInfo.LAYER_NAME, ProbeInfo.FORWARD_INDEX]

    # fields where we might expect different values
    compare = [ProbeInfo.INPUT_SHAPE, ProbeInfo.INPUT_TENSOR, ProbeInfo.OUTPUT_TENSOR]

    # print the comparison of summary1 and summary2
    summary1.compare_to(summary2, common, compare)


def backward_compare():
    model1 = MODEL(pretrained=True)
    model2 = MODEL(pretrained=True)

    dummy_input = imagenet_input()
    loss_func = nn.CrossEntropyLoss()
    dummy_target = torch.tensor([1])

    # two optimizer objects because they have internal state, e.g. update their momentum
    optimizer1 = torch.optim.SGD(model1.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)
    optimizer2 = torch.optim.SGD(model1.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)

    summary1 = probe_training(model1, dummy_input, optimizer1, loss_func, dummy_target)
    summary2 = probe_training(model2, dummy_input, optimizer2, loss_func, dummy_target)

    # fields that should for sure be the same
    common = [ProbeInfo.LAYER_NAME]

    # fields where we might expect different values
    compare = [ProbeInfo.INPUT_TENSOR, ProbeInfo.OUTPUT_TENSOR, ProbeInfo.GRAD_INPUT_TENSOR,
               ProbeInfo.GRAD_OUTPUT_TENSOR]

    # print the comparison of summary1 and summary2
    summary1.compare_to(summary2, common, compare)


def deterministic_backward_compare():
    dummy_target = torch.tensor([1])
    dummy_input = imagenet_input()
    loss_func = nn.CrossEntropyLoss()

    set_deterministic()
    model1 = MODEL(pretrained=True)
    optimizer1 = torch.optim.SGD(model1.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)
    summary1 = probe_training(model1, dummy_input, optimizer1, loss_func, dummy_target)

    set_deterministic()
    model2 = MODEL(pretrained=True)
    optimizer2 = torch.optim.SGD(model1.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)
    summary2 = probe_training(model2, dummy_input, optimizer2, loss_func, dummy_target)

    # fields that should for sure be the same
    common = [ProbeInfo.LAYER_NAME]

    # fields where we might expect different values
    compare = [ProbeInfo.INPUT_TENSOR, ProbeInfo.OUTPUT_TENSOR, ProbeInfo.GRAD_INPUT_TENSOR,
               ProbeInfo.GRAD_OUTPUT_TENSOR]

    # print the comparison of summary1 and summary2
    summary1.compare_to(summary2, common, compare)


if __name__ == '__main__':
    # summary()
    # print('\n\n\n')
    # forward_compare()
    # print('\n\n\n')
    # backward_compare()
    print('\n\n\n')
    deterministic_backward_compare()

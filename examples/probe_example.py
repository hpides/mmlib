import torch
from torch import nn
from torchvision import models

from mmlib.deterministic import set_deterministic
from mmlib.model_equals import imagenet_input
from mmlib.probe import ProbeInfo, probe_inference, probe_training


def summary():
    model = models.vgg19(pretrained=True)
    summary_info = [ProbeInfo.LAYER_NAME, ProbeInfo.INPUT_SHAPE, ProbeInfo.INPUT_TENSOR, ProbeInfo.OUTPUT_SHAPE,
                    ProbeInfo.OUTPUT_TENSOR]
    dummy_input = imagenet_input()

    # generate the summary using a dummy input
    summary = probe_inference(model, dummy_input)

    summary.print_summary(summary_info)

# def forward_compare():
#     model1 = models.vgg19(pretrained=True)
#     model2 = models.vgg19(pretrained=True)
#
#     dummy_input = imagenet_input()
#
#     summary1 = probe_inference(model1, dummy_input)
#     summary2 = probe_inference(model2, dummy_input)
#
#     # fields that should for sure be the same
#     common = [ProbeInfo.LAYER]
#
#     # fields where we might expect different values
#     compare = [ProbeInfo.INPUT_HASH, ProbeInfo.OUTPUT_HASH]
#
#     # print the comparison of summary1 and summary2
#     compare_summaries(summary1, summary2, compare, common=common)


# def backward_compare():
#     model1 = models.vgg19(pretrained=True)
#     model2 = models.vgg19(pretrained=True)
#
#     dummy_input = imagenet_input()
#     loss_func = nn.CrossEntropyLoss()
#     dummy_target = torch.tensor([1])
#
#     # two optimizer objects because they have internal state, e.g. update their momentum
#     optimizer1 = torch.optim.SGD(model1.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)
#     optimizer2 = torch.optim.SGD(model1.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)
#
#     summary1 = probe_training(model1, dummy_input, optimizer1, loss_func, dummy_target)
#     summary2 = probe_training(model2, dummy_input, optimizer2, loss_func, dummy_target)
#
#     # fields that should for sure be the same
#     common = [ProbeInfo.LAYER]
#
#     # fields where we might expect different values
#     compare = [ProbeInfo.INPUT_HASH, ProbeInfo.OUTPUT_HASH, ProbeInfo.GRAD_INPUT_HASH, ProbeInfo.GRAD_OUTPUT_HASH]
#
#     # print the comparison of summary1 and summary2
#     compare_summaries(summary1, summary2, compare, common=common)
#
#
# def deterministic_backward_compare():
#     model1 = models.vgg19(pretrained=True)
#     model2 = models.vgg19(pretrained=True)
#
#     dummy_input = imagenet_input()
#     loss_func = nn.CrossEntropyLoss()
#     dummy_target = torch.tensor([1])
#
#     # two optimizer objects because they have internal state, e.g. update their momentum
#     optimizer1 = torch.optim.SGD(model1.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)
#     optimizer2 = torch.optim.SGD(model1.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)
#
#     # in this case the important bit is that the seed are set before execution
#     set_deterministic()
#     summary1 = probe_training(model1, dummy_input, optimizer1, loss_func, dummy_target)
#     # reset seeds
#     set_deterministic()
#     summary2 = probe_training(model2, dummy_input, optimizer2, loss_func, dummy_target)
#
#     # fields that should for sure be the same
#     common = [ProbeInfo.LAYER]
#
#     # fields where we might expect different values
#     compare = [ProbeInfo.INPUT_HASH, ProbeInfo.OUTPUT_HASH, ProbeInfo.GRAD_INPUT_HASH, ProbeInfo.GRAD_OUTPUT_HASH]
#
#     # print the comparison of summary1 and summary2
#     compare_summaries(summary1, summary2, compare, common=common)


if __name__ == '__main__':
    summary()
    print('\n\n\n')
    # forward_compare()
    # print('\n\n\n')
    # backward_compare()
    # print('\n\n\n')
    # deterministic_backward_compare()

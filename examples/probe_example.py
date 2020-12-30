import torch
from torch import nn
from torchvision import models

from mmlib.model_equals import imagenet_input
from mmlib.probe import ProbeInfo, probe_inference, print_summary, probe_training, compare_summaries

if __name__ == '__main__':
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
            summary1 = probe_inference(model1, tensor1)
            summary2 = probe_inference(model2, tensor1)
            print_summary(summary1, output_info)
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
            sum_train1 = probe_training(model1, inp=tensor1, optimizer=optimizer, loss_func=loss_func,
                                        target=dummy_target)
            sum_train2 = probe_training(model2, inp=tensor1, optimizer=optimizer, loss_func=loss_func,
                                        target=dummy_target)
            compare_summaries(sum_train1, sum_train2,
                              [ProbeInfo.INPUT_HASH, ProbeInfo.OUTPUT_HASH, ProbeInfo.GRAD_INPUT_HASH,
                               ProbeInfo.GRAD_OUTPUT_HASH],
                              common=[ProbeInfo.LAYER, ProbeInfo.INPUT_SHAPE])

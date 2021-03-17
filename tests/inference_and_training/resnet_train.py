import torch

from mmlib.deterministic import set_deterministic
from schema.restorable_object import TrainService, OptimizerWrapper


class ResnetTrainService(TrainService):
    def train(self, model: torch.nn.Module, number_batches=None):

        set_deterministic()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        train_loader = self._get_dataloader()

        model.to(device)

        # switch to train mode
        model.train()

        for i, (images, target) in enumerate(train_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = torch.nn.CrossEntropyLoss()(output, target)

            # compute gradient and do SGD step
            # TODO restore manually here since optimizer needs model params
            optimizer = self._get_optimizer(model.parameters())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not (number_batches is None or i < number_batches - 1):
                break

    def _get_dataloader(self):
        dataloader = self.state_objs['dataloader'].instance
        return dataloader

    def _get_optimizer(self, parameters):
        optimizer_wrapper: OptimizerWrapper = self.state_objs['optimizer']

        if not optimizer_wrapper.instance:
            optimizer_wrapper.restore_instance({'params': parameters})

        return optimizer_wrapper.instance

import time

import torch
from schema.inference_info import InferenceService


class ResnetInferenceService(InferenceService):

    def infer(self, model: torch.nn.Module):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        val_loader = self._get_dataloader()

        # load model to device
        model = model.to(device)

        # switch to evaluate mode
        model.eval()

        outputs = []

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                images = images.to(device)

                # compute output
                output = model(images)

            return outputs

    def _get_dataloader(self):
        dataloader = self.state_objs['data_loader'].instace
        return dataloader

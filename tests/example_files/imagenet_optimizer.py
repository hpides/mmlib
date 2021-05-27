import torch
from torch.optim import SGD

from schema.restorable_object import StateFileRestorableObject


class ImagenetOptimizer(SGD, StateFileRestorableObject):

    def save_instance_state(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def restore_instance_state(self, path):
        self.load_state_dict(torch.load(path))

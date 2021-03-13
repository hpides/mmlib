import torch
from torch.utils.data import DataLoader

from schema.restorable_object import RestorableObjectWrapper


class WrappedDataLoader(RestorableObjectWrapper):

    def _save_instance_state(self):
        dataloader: torch.utils.data.DataLoader = None

    def _restore_instance_from_state(self):
        pass

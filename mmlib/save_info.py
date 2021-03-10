import torch


class ModelSaveInfo:

    def __init__(self, model: torch.nn.Module, base_model: str, code: str, codename: str, recover_val: bool,
                 dummy_input_shape: [int]):
        self.model = model
        self.base_model = base_model
        self.code = code
        self.codename = codename
        self.recover_val = recover_val
        self.dummy_input_shape = dummy_input_shape

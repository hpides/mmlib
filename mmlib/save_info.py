import torch

from schema.environment import Environment
from schema.restorable_object import RestorableObjectWrapper


class InferenceSaveInfo:
    def __init__(self, dataloader: RestorableObjectWrapper, pre_processor: RestorableObjectWrapper, environment: Environment):
        self.dataloader = dataloader
        self.pre_processor = pre_processor
        self.environment = environment


class ModelSaveInfo:

    def __init__(self, model: torch.nn.Module, base_model: str, code: str, class_name: str, recover_val: bool,
                 dummy_input_shape: [int], inference_info: InferenceSaveInfo):
        self.model = model
        self.base_model = base_model
        self.code = code
        self.class_name = class_name
        self.recover_val = recover_val
        self.dummy_input_shape = dummy_input_shape
        self.inference_info = inference_info

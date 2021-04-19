import torch

from schema.environment import Environment
from schema.restorable_object import StateDictRestorableObjectWrapper
from util.helper import class_name, source_file


class TrainSaveInfo:
    def __init__(self, train_service_wrapper: StateDictRestorableObjectWrapper, train_kwargs: dict,
                 environment: Environment):
        self.train_service = train_service_wrapper.instance
        self.train_wrapper_code = source_file(train_service_wrapper)
        self.train_wrapper_class_name = class_name(train_service_wrapper)
        self.train_kwargs = train_kwargs
        self.environment = environment


class ModelSaveInfo:
    def __init__(self, model: torch.nn.Module, base_model: str, dummy_input_shape: [int], model_code: str = None):
        self.model = model
        self.base_model = base_model
        self.dummy_input_shape = dummy_input_shape
        if model:
            self.model_class_name = class_name(model)
            self.model_code = model_code if model_code else source_file(model)


class ProvModelSaveInfo(ModelSaveInfo):
    def __init__(self, model: torch.nn.Module, base_model: str, model_code: str, dummy_input_shape: [int],
                 raw_dataset: str, train_info: TrainSaveInfo):
        super().__init__(model, base_model, model_code, dummy_input_shape)
        self.raw_dataset = raw_dataset
        self.train_info = train_info

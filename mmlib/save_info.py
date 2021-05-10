import torch

from schema.environment import Environment
from schema.restorable_object import StateDictRestorableObjectWrapper
from util.helper import class_name, source_file


class TrainSaveInfo:
    def __init__(self, train_service_wrapper: StateDictRestorableObjectWrapper, train_kwargs: dict):
        self.train_service = train_service_wrapper.instance
        self.train_wrapper_code = source_file(train_service_wrapper)
        self.train_wrapper_class_name = class_name(train_service_wrapper)
        self.train_kwargs = train_kwargs


class ModelSaveInfo:
    def __init__(self, model: torch.nn.Module, base_model: str, dummy_input_shape: [int], environment: Environment,
                 model_code: str = None):
        self.model = model
        self.base_model = base_model
        # TODO can we remove dummy inout shape?
        self.dummy_input_shape = dummy_input_shape
        self.environment = environment
        if model:
            self.model_class_name = class_name(model)
            self.model_code = model_code if model_code else source_file(model)


class ProvModelSaveInfo(ModelSaveInfo):
    def __init__(self, model: torch.nn.Module, base_model: str, model_code: str, dummy_input_shape: [int],
                 raw_dataset: str, train_info: TrainSaveInfo, environment: Environment):
        super().__init__(model, base_model, dummy_input_shape, environment, model_code)
        self.raw_dataset = raw_dataset
        self.train_info = train_info

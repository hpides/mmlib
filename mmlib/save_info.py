import torch

from mmlib.schema import Environment
from mmlib.schema.restorable_object import StateDictRestorableObjectWrapper
from mmlib.util.helper import class_name, source_file


class TrainSaveInfo:
    def __init__(self, train_service_wrapper: StateDictRestorableObjectWrapper, train_kwargs: dict):
        self.train_service = train_service_wrapper.instance
        self.train_wrapper_code = source_file(train_service_wrapper)
        self.train_wrapper_class_name = class_name(train_service_wrapper)
        self.train_kwargs = train_kwargs


class ModelSaveInfo:
    def __init__(self, model: torch.nn.Module, base_model: str, environment: Environment, model_code: str = None):
        self.model = model
        self.base_model = base_model
        self.environment = environment
        if model:
            self.model_class_name = class_name(model)
            self.model_code = model_code if model_code else source_file(model)


class ProvModelSaveInfo(ModelSaveInfo):
    def __init__(self, model: torch.nn.Module, base_model: str, model_code: str, raw_dataset: str,
                 train_info: TrainSaveInfo, environment: Environment):
        super().__init__(model, base_model, environment, model_code)
        self.raw_dataset = raw_dataset
        self.train_info = train_info

import torch

from schema.environment import Environment
from schema.restorable_object import RestorableObjectWrapper, StateDictObj


class TrainSaveInfo:
    def __init__(self, train_service: StateDictObj, train_service_code: str, train_service_class_name: str,
                 train_kwargs: dict, train_wrapper_code: str, train_wrapper_class_name: str, environment: Environment):
        self.train_service = train_service
        self.train_service_code = train_service_code
        self.train_service_class_name = train_service_class_name
        self.train_wrapper_code = train_wrapper_code
        self.train_wrapper_class_name = train_wrapper_class_name
        self.train_kwargs = train_kwargs
        self.environment = environment


class ProvRecoverInfo:
    def __init__(self, raw_dataset: str, model_code: str, model_class_name: str, train_info: TrainSaveInfo):
        self.raw_dataset = raw_dataset
        self.model_code = model_code
        self.model_class_name = model_class_name
        self.train_info = train_info


class InferenceSaveInfo:
    def __init__(self, data_wrapper: RestorableObjectWrapper, dataloader: RestorableObjectWrapper,
                 pre_processor: RestorableObjectWrapper, environment: Environment):
        self.data_wrapper = data_wrapper
        self.dataloader = dataloader
        self.pre_processor = pre_processor
        self.environment = environment


class ModelSaveInfo:
    def __init__(self, model: torch.nn.Module, base_model: str, code: str, class_name: str, recover_val: bool,
                 dummy_input_shape: [int], inference_info: InferenceSaveInfo, prov_rec_info: ProvRecoverInfo):
        self.model = model
        self.base_model = base_model
        self.code = code
        self.class_name = class_name
        self.recover_val = recover_val
        self.dummy_input_shape = dummy_input_shape
        self.inference_info = inference_info
        self.prov_rec_info = prov_rec_info

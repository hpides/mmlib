import torch

from schema.RestorableObject import RestorableObject
from schema.dataset import Dataset


class ModelSaveInfo:
    pass

class ModelRestoreInfo:
    pass


class RecoverValInfo:

    def __init__(self, recover_val: bool = False, dummy_input_shape: [int] = None):
        self.recover_val = recover_val
        self.dummy_input_shape = dummy_input_shape


class RestorableObjectSaveInfo:

    def __init__(self, code: str, class_name: str, state: str):
        self.code = code
        self.class_name = class_name
        self.state = state


class DataLoaderSavaInfo(RestorableObjectSaveInfo):

    def __init__(self, code: str, class_name: str, state: str):
        super().__init__(code, class_name, state)


class TrainInfo:

    def __init__(self, data_loader_code: str, data_loader,
                 pre_processor: RestorableObject, dataset: Dataset):
        # TODO fix adding info for schema objects
        self.data_loader = data_loader
        self.pre_processor = pre_processor


class FullModelSaveInfo(RecoverValInfo):

    def __init__(self, model: torch.nn.Module, code: str, code_name: str, recover_val: bool = False,
                 dummy_input_shape: [int] = None):
        super().__init__(recover_val, dummy_input_shape)
        self.model = model
        self.code = code
        self.code_name = code_name

        # TODO need to add inference and train info etc.


class FullModelVersionSaveInfo(RecoverValInfo):

    def __init__(self, model: torch.nn.Module, base_model_id: str, recover_val: bool = False,
                 dummy_input_shape: [int] = None):
        super().__init__(recover_val, dummy_input_shape)
        self.model = model
        self.base_model_id = base_model_id
        self.recover_val = recover_val
        self.dummy_input_shape = dummy_input_shape
        # TODO need to add inference and train info etc.


class ProvenanceModelVersionSaveInfo:

    def __init__(self, base_model_id: str,
                 dataset_code: str, dataset_class_name: str, raw_data_root: str,  # for dataset
                 # TODO train_recover_info
                 # TODO train environment
                 # TODO optional inference_info
                 model: torch.nn.Module, recover_val: bool = False, dummy_input_shape: [int] = None  # for recover_val
                 ):
        self.base_model_id = base_model_id
        self.dataset_code = dataset_code
        self.dataset_class_name = dataset_class_name
        self.raw_data_root = raw_data_root
        self.model = model
        self.recover_val = recover_val
        self.dummy_input_shape = dummy_input_shape

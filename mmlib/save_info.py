import torch


class FullModelSafeInfo:

    def __init__(self, model: torch.nn.Module, code: str, code_name: str, recover_val: bool = False,
                 dummy_input_shape: [int] = None):
        self.model = model
        self.code = code
        self.code_name = code_name
        self.recover_val = recover_val
        self.dummy_input_shape = dummy_input_shape
        # TODO need to add inference and train info etc.


class FullModelVersionSafeInfo:

    def __init__(self, model: torch.nn.Module, base_model_id: str, recover_val: bool = False,
                 dummy_input_shape: [int] = None):
        self.model = model
        self.base_model_id = base_model_id
        self.recover_val = recover_val
        self.dummy_input_shape = dummy_input_shape
        # TODO need to add inference and train info etc.


class ProvenanceModelVersionSafeInfo:

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

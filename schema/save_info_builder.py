import torch

from mmlib.save_info import FullModelSafeInfo


class RecoverValInfoBuilder:

    def __init__(self):
        self.recover_val = False
        self.dummy_input_shape = None

    def add_recover_val(self, dummy_input_shape: [int] = None):
        """
        Indicates that recover validation info should be saved and adds the required info.
        :param dummy_input_shape: The shape of the dummy input that should be used to produce an inference result.
        """
        self.recover_val = True
        self.dummy_input_shape = dummy_input_shape


class FullModelSafeInfoBuilder(RecoverValInfoBuilder):

    def __init__(self):
        super().__init__()
        self.model = None
        self.code = None
        self.code_name = None

    def add_model_info(self, model: torch.nn.Module, code: str, code_name: str):
        """
        Adds the general model information
        :param model: The actual model to save as an instance of torch.nn.Module.
        :param code: The path to the code of the model (is needed for recover process).
        :param code_name: The name of the model, i.e. the model constructor (is needed for recover process).
        """
        self.model = model
        self.code = code
        self.code_name = code_name

    def build_full_model_save_info(self) -> FullModelSafeInfo:
        # TODO check if all info is available
        safe_info = FullModelSafeInfo(self.model, self.code, self.code_name, self.recover_val, self.dummy_input_shape)
        return safe_info

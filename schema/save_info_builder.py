import torch

from mmlib.save_info import FullModelSafeInfo


class FullModelSafeInfoBuilder:

    def __init__(self):
        self.model = None
        self.code = None
        self.code_name = None
        self.recover_val = False
        self.dummy_input_shape = None
    """TODO use comments
           Saves a model together with the given metadata.
           :param model: The actual model to save as an instance of torch.nn.Module.
           :param code: The path to the code of the model (is needed for recover process).
           :param code_name: The name of the model, i.e. the model constructor (is needed for recover process).
           :param recover_val: Indicates if along with the model itself also information is stored to later validate that
           restoring the model lead to the exact same model. It is checked by comparing the model weights and the inference
           result on dummy input. If this flag is true, a dummy_input_shape has to be provided.
           :param dummy_input_shape: The shape of the dummy input that should be used to produce an inference result.
           :return: Returns the id that was used to store the model.
           """

    def add_model_info(self, model: torch.nn.Module, code: str, code_name: str):
        self.model = model
        self.code = code
        self.code_name = code_name

    def add_recover_val(self, dummy_input_shape: [int] = None):
        self.recover_val = True
        self.dummy_input_shape = dummy_input_shape

    def build_full_model_save_info(self) -> FullModelSafeInfo:
        # TODO check if all info is available
        safe_info = FullModelSafeInfo(self.model, self.code, self.code_name, self.recover_val, self.dummy_input_shape)
        return safe_info

import torch

from mmlib.save_info import FullModelSafeInfo, FullModelVersionSafeInfo


class RecoverValInfoBuilder:

    def __init__(self):
        self._recover_val = False
        self._dummy_input_shape = None

    def add_recover_val(self, dummy_input_shape: [int] = None):
        """
        Indicates that recover validation info should be saved and adds the required info.
        :param dummy_input_shape: The shape of the dummy input that should be used to produce an inference result.
        """
        self._recover_val = True
        self._dummy_input_shape = dummy_input_shape


class FullModelSafeInfoBuilder(RecoverValInfoBuilder):

    def __init__(self):
        super().__init__()
        self._model = None
        self._code = None
        self._code_name = None

    def add_model_info(self, model: torch.nn.Module, code: str, code_name: str):
        """
        Adds the general model information
        :param model: The actual model to save as an instance of torch.nn.Module.
        :param code: The path to the code of the model (is needed for recover process).
        :param code_name: The name of the model, i.e. the model constructor (is needed for recover process).
        """
        self._model = model
        self._code = code
        self._code_name = code_name

    def build(self) -> FullModelSafeInfo:
        # TODO check if all info is available
        safe_info = FullModelSafeInfo(self._model, self._code, self._code_name, self._recover_val,
                                      self._dummy_input_shape)
        return safe_info


class FullModelVersionSafeInfoBuilder(RecoverValInfoBuilder):

    def __init__(self):
        super().__init__()
        self._model = None
        self._base_model_id = None

    def add_model_version_info(self, model: torch.nn.Module, base_model_id: str):
        """
        :param model: The actual model to save as an instance of torch.nn.Module.
        :param base_model_id: the model id of the base_model.
        """
        self._model = model
        self._base_model_id = base_model_id

    def build(self) -> FullModelVersionSafeInfo:
        version_info = FullModelVersionSafeInfo(self._model, self._base_model_id, self._recover_val,
                                                self._dummy_input_shape)
        return version_info

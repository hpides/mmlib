import torch

from mmlib.save_info import ModelSaveInfo, InferenceSaveInfo
from schema.environment import Environment
from schema.restorable_object import RestorableObjectWrapper


class ModelSaveInfoBuilder:

    def __init__(self):
        super().__init__()
        self._model = None
        self._base_model = None
        self._code = None
        self._code_name = None
        self._recover_val = False
        self._dummy_input_shape = None
        self._inference_data_wrapper = None
        self._inference_dataloader = None
        self._inference_pre_processor = None
        self._inference_environment = None

    def add_model_info(self, model: torch.nn.Module, code: str = None, model_class_name: str = None,
                       base_model_id: str = None):
        """
        Adds the general model information
        :param model: The actual model to save as an instance of torch.nn.Module.
        :param code: (only required if base model not given) The path to the code of the model
        (is needed for recover process).
        :param model_class_name: (only required if base model not given) The name of the model, i.e. the model constructor (is needed for recover process).
        :param base_model_id: The id of the base model.
        """
        self._model = model
        self._base_model = base_model_id
        self._code = code
        self._code_name = model_class_name

    def add_recover_val(self, dummy_input_shape: [int] = None):
        """
        Indicates that recover validation info should be saved and adds the required info.
        :param dummy_input_shape: The shape of the dummy input that should be used to produce an inference result.
        """
        self._recover_val = True
        self._dummy_input_shape = dummy_input_shape

    def add_inference_info(self, data_wrapper: RestorableObjectWrapper, dataloader: RestorableObjectWrapper,
                           pre_processor: RestorableObjectWrapper, environment: Environment):
        """
        Indicates that inference info should be saved and adds the required info.
        :param data_wrapper: The data_wrapper wrapped in an RestorableObject.
        :param dataloader: The dataloader wrapped in an RestorableObject.
        :param pre_processor: The pre_processor wrapped in an RestorableObject.
        :param environment: The environment as an object of type Environment.
        """
        self._inference_data_wrapper = data_wrapper
        self._inference_dataloader = dataloader
        self._inference_pre_processor = pre_processor
        self._inference_environment = environment

    def build(self) -> ModelSaveInfo:
        inf_info = None
        if self._inference_data_wrapper or self._inference_dataloader or\
                self._inference_pre_processor or self._inference_environment:

            assert self._inference_data_wrapper, 'if inference info shall be stored -> data wrapper must be given'
            assert self._inference_dataloader, 'if inference info shall be stored -> data loader must be given'
            assert self._inference_environment, 'if inference info shall be stored -> environment must be given'

            inf_info = InferenceSaveInfo(data_wrapper=self._inference_data_wrapper,
                                         dataloader=self._inference_dataloader,
                                         pre_processor=self._inference_pre_processor,
                                         environment=self._inference_environment)

        save_info = ModelSaveInfo(self._model, self._base_model, self._code, self._code_name, self._recover_val,
                                  self._dummy_input_shape, inference_info=inf_info)
        return save_info

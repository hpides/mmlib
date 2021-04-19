import torch

from mmlib.save_info import ModelSaveInfo, TrainSaveInfo, ProvModelSaveInfo
from schema.environment import Environment
from schema.restorable_object import StateDictObj


class ModelSaveInfoBuilder:

    def __init__(self):
        super().__init__()
        self._model = None
        self._base_model = None
        self._code = None
        self._dummy_input_shape = None
        self._prov_raw_data = None
        self._prov_env = None
        self._prov_train_service = None
        self._prov_train_kwargs = None
        self._prov_train_wrapper_code = None
        self._prov_train_wrapper_class_name = None

        self.general_model_info_added = False
        self.prov_model_info_added = False

    def add_model_info(self, model: torch.nn.Module = None, code: str = None, base_model_id: str = None):
        """
        Adds the general model information
        :param model: The actual model to save as an instance of torch.nn.Module.
        :param code: (only required if base model not given or if it can not be automatically inferred) The path to the
         code of the model .
        constructor (is needed for recover process).
        :param base_model_id: The id of the base model.
        """
        self._model = model
        self._base_model = base_model_id
        self._code = code

        self.general_model_info_added = True

    def add_prov_data(self, raw_data_path: str, env: Environment, train_service: StateDictObj, train_kwargs: dict,
                      wrapper_code: str, wrapper_class_name: str):
        """
        Adds information that is required to store a model using its provenance data.
        :param raw_data_path: The path to the raw data that was used as the dataset.
        :param env: The environment the training was/will be performed in.
        :param train_service: The train service that was/will be used to train the model.
        :param train_kwargs: The kwargs that will be given to the train method of the train service.
        :param wrapper_code: The path to the code for the train service wrapper.
        :param wrapper_class_name: The class name of the train service wrapper.
        """
        self._prov_raw_data = raw_data_path
        self._prov_env = env
        self._prov_train_service = train_service
        self._prov_train_kwargs = train_kwargs
        self._prov_train_wrapper_code = wrapper_code
        self._prov_train_wrapper_class_name = wrapper_class_name

        self.prov_model_info_added = True

    def build(self) -> ModelSaveInfo:

        if self.general_model_info_added:
            assert self._valid_baseline_save_model_info(), 'info not sufficient'
            if self.prov_model_info_added:
                assert self._valid_prov_save_model_info(), 'info not sufficient'
                return self._build_prov_save_info()
            else:
                return self._build_baseline_save_info()

    def _build_baseline_save_info(self):
        save_info = ModelSaveInfo(
            model=self._model,
            base_model=self._base_model,
            model_code=self._code,
            dummy_input_shape=self._dummy_input_shape)

        return save_info

    def _build_prov_save_info(self):
        prov_train_info = TrainSaveInfo(
            train_service=self._prov_train_service,
            train_wrapper_code=self._prov_train_wrapper_code,
            train_wrapper_class_name=self._prov_train_wrapper_class_name,
            train_kwargs=self._prov_train_kwargs,
            environment=self._prov_env)

        save_info = ProvModelSaveInfo(
            model=self._model,
            base_model=self._base_model,
            model_code=self._code,
            dummy_input_shape=self._dummy_input_shape,
            raw_dataset=self._prov_raw_data,
            train_info=prov_train_info)

        return save_info

    def _valid_baseline_save_model_info(self):
        return self._model or self._base_model

    def _valid_prov_save_model_info(self):
        return self._valid_baseline_save_model_info() and self._base_model \
               and self._prov_raw_data and self._prov_env and self._prov_train_service \
               and self._prov_train_kwargs and self._prov_train_kwargs \
               and self._prov_train_wrapper_code and self._prov_train_wrapper_class_name

import torch

from mmlib.save_info import ModelSaveInfo, TrainSaveInfo, ProvModelSaveInfo
from mmlib.schema.environment import Environment
from mmlib.schema.restorable_object import StateDictRestorableObjectWrapper


class ModelSaveInfoBuilder:

    def __init__(self):
        super().__init__()
        self._model = None
        self._base_model = None
        self._code = None
        self._prov_raw_data = None
        self._env = None
        self._prov_train_kwargs = None
        self._prov_train_service_wrapper = None

        self.general_model_info_added = False
        self.prov_model_info_added = False

    def add_model_info(self, env: Environment, model: torch.nn.Module = None, code: str = None,
                       base_model_id: str = None):
        """
        Adds the general model information
        :param env: The environment the training was/will be performed in.
        :param model: The actual model to save as an instance of torch.nn.Module.
        :param code: (only required if base model not given or if it can not be automatically inferred) The path to the
         code of the model .
        constructor (is needed for recover process).
        :param base_model_id: The id of the base model.
        """
        self._env = env
        self._model = model
        self._base_model = base_model_id
        self._code = code
        self.general_model_info_added = True

    def add_prov_data(self, raw_data_path: str, train_kwargs: dict,
                      train_service_wrapper: StateDictRestorableObjectWrapper):
        """
        Adds information that is required to store a model using its provenance data.
        :param raw_data_path: The path to the raw data that was used as the dataset.
        :param train_kwargs: The kwargs that will be given to the train method of the train service.
        :param train_service_wrapper: The train service wrapper that wraps the train service used to train the model.
        """
        self._prov_raw_data = raw_data_path
        self._prov_train_kwargs = train_kwargs
        self._prov_train_service_wrapper = train_service_wrapper

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
            environment=self._env)

        return save_info

    def _build_prov_save_info(self):
        prov_train_info = TrainSaveInfo(
            train_service_wrapper=self._prov_train_service_wrapper,
            train_kwargs=self._prov_train_kwargs)

        save_info = ProvModelSaveInfo(

            model=self._model,
            base_model=self._base_model,
            model_code=self._code,
            raw_dataset=self._prov_raw_data,
            train_info=prov_train_info,
            environment=self._env)

        return save_info

    def _valid_baseline_save_model_info(self):
        return self._model or self._base_model

    def _valid_prov_save_model_info(self):
        return self._valid_baseline_save_model_info() and self._base_model \
               and self._prov_raw_data and self._env and self._prov_train_service_wrapper \
               and self._prov_train_kwargs and self._prov_train_kwargs

import abc
import os
import sys
import tempfile

import torch

from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from mmlib.save_info import ModelSaveInfo
# Future work, se if it would make sense to use protocol here
from schema.model_info import ModelInfo
from schema.recover_info import FullModelRecoverInfo
from schema.store_type import ModelStoreType

MODEL_WEIGHTS = 'model_weights'


class RestoredModelInfo:
    # TODO move to separate file

    def __init__(self, model: torch.nn.Module):
        self.model = model


class AbstractSaveService(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def save_model(self, model_save_info: ModelSaveInfo) -> str:
        """
        Saves a model together with the given metadata.
        :param model_save_info: An instance of ModelSaveInfo providing all the info needed to save the model.
        :return: Returns the id that was used to store the model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def recover_model(self, model_id: str, check_recover_val=False) -> RestoredModelInfo:
        """
        Recovers a the model and metadata identified by the given model id.
        :param model_id: The id to identify the model with.
        :param check_recover_val: The flag that indicates if the recover validation data (if there) is used to validate
        the restored model.
        :return: The recovered model and metadata bundled in an object of type ModelRestoreInfo.
        """


class BaselineSaveService(AbstractSaveService):
    """A Service that offers functionality to store PyTorch models by making use of a persistence service.
         The metadata is stored in JSON like dictionaries, files and weights are stored as files."""

    def __init__(self, file_pers_service: AbstractFilePersistenceService,
                 dict_pers_service: AbstractDictPersistenceService):
        """
        :param file_pers_service: An instance of AbstractFilePersistenceService that is used to store files.
        :param dict_pers_service: An instance of AbstractDictPersistenceService that is used to store metadata as dicts.
        """
        self._file_pers_service = file_pers_service
        self._dict_pers_service = dict_pers_service

    def save_model(self, model_save_info: ModelSaveInfo) -> str:
        self._check_consistency(model_save_info)

        # usually we would consider at this bit how we best store the given model
        # but since this is the baseline service we just store the full model every time.
        model_id = self._save_full_model(model_save_info)

        return model_id

    def recover_model(self, model_id: str, check_recover_val=False) -> RestoredModelInfo:
        # in this baseline approach we always store the full model (pickled weights + code)

        with tempfile.TemporaryDirectory() as tmp_path:
            model_info = ModelInfo.load(model_id, self._file_pers_service, self._dict_pers_service, tmp_path)

            # recover model form info
            recover_info: FullModelRecoverInfo = model_info.recover_info

            model = self._init_model(recover_info.model_code_file_path, recover_info.model_class_name)
            s_dict = self._recover_pickled_weights(recover_info.weights_file_path)
            model.load_state_dict(s_dict)

            restored_model_info = RestoredModelInfo(model)

        return restored_model_info

    def _check_consistency(self, model_save_info):
        if model_save_info.recover_val:
            assert model_save_info.dummy_input_shape, 'to store recover_val information a dummy input function needs ' \
                                                      'to be provided'

        # TODO add more/other checks

    def _save_full_model(self, model_save_info: ModelSaveInfo) -> str:
        # TODO check if recover val is true and implement

        with tempfile.TemporaryDirectory() as tmp_path:
            weights_path = self._pickle_weights(model_save_info.model, tmp_path)

            derived_from = model_save_info.base_model if model_save_info.base_model else None

            if derived_from and not(model_save_info.code or model_save_info.class_name):
                model_info = ModelInfo.load(derived_from, self._file_pers_service, self._dict_pers_service, tmp_path)
                recover_info: FullModelRecoverInfo = model_info.recover_info

                model_save_info.code = recover_info.model_code_file_path
                model_save_info.class_name = recover_info.model_class_name

            # if the model to store is not derived from another model code and class name have to me defined
            recover_info = FullModelRecoverInfo(weights_file_path=weights_path,
                                                model_code_file_path=model_save_info.code,
                                                model_class_name=model_save_info.class_name)

            model_info = ModelInfo(store_type=ModelStoreType.PICKLED_WEIGHTS, recover_info=recover_info,
                                   derived_from_id=derived_from)

            model_info_id = model_info.persist(self._file_pers_service, self._dict_pers_service)

        return model_info_id

    def _pickle_weights(self, model, save_path):
        # store pickle dump of model
        weight_path = os.path.join(save_path, MODEL_WEIGHTS)
        torch.save(model.state_dict(), weight_path)

        return weight_path

    def _init_model(self, code, generate_call):
        path, file = os.path.split(code)
        module = file.replace('.py', '')
        sys.path.append(path)
        exec('from {} import {}'.format(module, generate_call))
        model = eval('{}()'.format(generate_call))

        return model

    def _recover_pickled_weights(self, weights_file):
        state_dict = torch.load(weights_file)

        return state_dict

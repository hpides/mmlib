import abc
import os
import sys
from enum import Enum

import torch

from mmlib.persistence import AbstractPersistenceService
from mmlib.schema.recover_info_t1 import RecoverInfoT1
from util.helper import clean
from util.zip import zip_path, unzip

TMP_DIR = 'tmp-dir'

NAME = 'name'
MODELS = 'models'

ID = '_id'

MODEL_INFO = 'model_info'
RECOVER_T1 = 'recover_t1'


class SaveType(Enum):
    PICKLED_WEIGHTS = 1
    WEIGHT_UPDATES = 2
    PROVENANCE = 3


class ModelInfo(Enum):
    STORE_TYPE = 'store_type'
    RECOVER_INFO = 'recover_info'
    DERIVED_FROM = 'derived_from'
    INFERENCE_INFO = 'inference_info'
    TRAIN_INFO = 'train_info'


# TODO if for experiments Python 3.8 is available, use protocol here
class AbstractSaveRecoverService(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'save_model') and
                callable(subclass.save_model) and
                hasattr(subclass, 'save_version') and
                callable(subclass.save_version) and
                hasattr(subclass, 'recover_model') and
                callable(subclass.recover_model) and
                hasattr(subclass, 'saved_model_ids') and
                callable(subclass.saved_model_ids) or
                NotImplemented)

    @abc.abstractmethod
    def save_model(self, model: torch.nn.Module, code: str, code_name: str, ) -> str:
        """
        Saves a model together with the given metadata.
        :param model: The actual model to save as an instance of torch.nn.Module.
        :param code: The path to the code of the model (is needed for recover process).
        :param code_name: The name of the model, i.e. the model constructor (is needed for recover process).
        :return: Returns the id that was used to store the model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save_version(self, model: torch.nn.Module, base_model_id: str) -> str:
        """
        Saves a new model version by referring to the base_model.
        :param model: The actual model to save as an instance of torch.nn.Module.
        :param base_model_id: the model id of the base_model.
        :return: Returns the ID that was used to store the new model version data in the MongoDB.
        """

    @abc.abstractmethod
    def saved_model_ids(self) -> [str]:
        """Returns list of saved models ids"""
        raise NotImplementedError

    @abc.abstractmethod
    def saved_model_infos(self) -> [dict]:
        """Returns list of saved models infos"""
        raise NotImplementedError

    @abc.abstractmethod
    def recover_model(self, model_id: str) -> torch.nn.Module:
        """
        Recovers a the model identified by the given model id.
        :param model_id: The id to identify the model with.
        :return: The recovered model as an object of type torch.nn.Module.
        """

    @abc.abstractmethod
    def model_save_size(self, model_id: str) -> float:
        """
        Calculates and returns the amount of bytes that are used for storing the model.
        :param model_id: The id to identify the model.
        :return: The amount of bytes used to store the model.
        """
        raise NotImplementedError


class SimpleSaveRecoverService(AbstractSaveRecoverService):
    """A Service that offers functionality to store PyTorch models by making use of a persistence service.
     The metadata is stored in JSON like dictionaries, files and weights are stored as files."""

    def __init__(self, persistence_service: AbstractPersistenceService, tmp_path: str):
        """
        :param persistence_service: An instance of AbstractPersistenceService that is used to store metadata and files.
        :param tmp_path: A path/directory that can be used to store files temporarily.
        """
        self._pers_service = persistence_service
        self._tmp_path = os.path.abspath(tmp_path)

    def save_model(self, model: torch.nn.Module, code: str, code_name: str, ) -> str:
        recover_info_t1 = self._save_model_t1(model, code, code_name)
        recover_info_id = self._pers_service.save_dict(recover_info_t1.to_dict(), RECOVER_T1)

        # TODO to implement other fields that are default None
        model_id = self._save_model_info(SaveType.PICKLED_WEIGHTS.value, recover_info_id)

        return model_id

    def _save_model_info(self, save_type, recover_info_id, derived_from=None, inference_info=None, train_info=None):
        model_dict = {
            ModelInfo.STORE_TYPE.value: save_type,
            ModelInfo.RECOVER_INFO.value: recover_info_id,
            ModelInfo.DERIVED_FROM.value: derived_from,
            ModelInfo.INFERENCE_INFO.value: inference_info,
            ModelInfo.TRAIN_INFO.value: train_info

        }
        model_id = self._pers_service.save_dict(model_dict, MODEL_INFO)
        return model_id

    def _save_model_t1(self, model, code, code_name):
        gen_id = self._pers_service.generate_id()
        dst_path = os.path.join(self._tmp_path, gen_id)

        zip_file = self._pickle_weights(model, dst_path)
        zip_file_id = self._pers_service.save_file(zip_file)
        code_file_id = self._pers_service.save_file(code)
        clean(dst_path)
        clean(zip_file)

        recover_info_t1 = RecoverInfoT1(r_id=gen_id, weights=zip_file_id, model_code=code_file_id, code_name=code_name)

        return recover_info_t1

    def save_version(self, model: torch.nn.Module, base_model_id: str) -> str:
        base_model_info = self._pers_service.recover_dict(base_model_id, MODEL_INFO)
        base_model_recover_info = self._get_recover_info_t1(base_model_info)

        # copy fields from previous model that will stay the same
        code_name = base_model_recover_info.code_name

        tmp_path = os.path.abspath(os.path.join(self._tmp_path, TMP_DIR))
        os.mkdir(tmp_path)  # TODO maybe use with context
        code = self._pers_service.recover_file(base_model_recover_info.model_code, tmp_path)

        recover_info_t1 = self._save_model_t1(model, code, code_name)
        clean(tmp_path)

        recover_info_id = self._pers_service.save_dict(recover_info_t1.to_dict(), RECOVER_T1)

        # TODO to implement other fields that are default None
        model_id = self._save_model_info(SaveType.PICKLED_WEIGHTS.value, recover_info_id, derived_from=base_model_id)

        return model_id

    def saved_model_ids(self) -> [str]:
        pass
        # str_ids = list(map(str, self._mongo_service.get_ids()))
        # return str_ids

    def saved_model_infos(self) -> [dict]:
        pass

    def model_save_size(self, model_id: str) -> float:
        pass
        # model_id = bson.ObjectId(model_id)
        #
        # document_size = self._mongo_service.document_size(model_id)
        #
        # meta_data = self._mongo_service.get_dict(model_id)
        # save_path = meta_data[SAVE_PATH]
        # zip_size = os.path.getsize(save_path)
        #
        # return document_size + zip_size

    def recover_model(self, model_id: str) -> torch.nn.Module:
        model_info = self._pers_service.recover_dict(model_id, MODEL_INFO)
        recover_info_t1 = self._get_recover_info_t1(model_info)
        weights_file_id = recover_info_t1.weights

        tmp_path = os.path.abspath(os.path.join(self._tmp_path, TMP_DIR))
        os.mkdir(tmp_path)  # TODO maybe use with context
        code_id = recover_info_t1.model_code
        code = self._pers_service.recover_file(code_id, tmp_path)
        generate_call = recover_info_t1.code_name
        model = self._init_model(code, generate_call)

        weights_file = self._pers_service.recover_file(weights_file_id, tmp_path)
        s_dict = self._recover_pickled_weights(weights_file, tmp_path)
        model.load_state_dict(s_dict)

        clean(tmp_path)

        return model

    def _get_recover_info_t1(self, model_info):
        recover_info_id = model_info[ModelInfo.RECOVER_INFO.value]
        recover_info_dict = self._pers_service.recover_dict(recover_info_id, RECOVER_T1)

        recover_info = RecoverInfoT1()
        recover_info.load_dict(recover_info_dict)

        return recover_info

    def _recover_pickled_weights(self, weights_file, extract_path):
        unpacked_path = unzip(weights_file, extract_path)
        pickle_path = os.path.join(unpacked_path, 'model_weights')
        state_dict = torch.load(pickle_path)

        return state_dict

    def _pickle_weights(self, model, save_path):
        # create directory to store in
        abs_save_path = os.path.abspath(save_path)
        os.makedirs(abs_save_path)

        # store pickle dump of model
        torch.save(model.state_dict(), os.path.join(abs_save_path, 'model_weights'))

        # zip everything
        return zip_path(save_path)

    def _init_model(self, code, generate_call):
        path, file = os.path.split(code)
        module = file.replace('.py', '')
        sys.path.append(path)
        exec('from {} import {}'.format(module, generate_call))
        model = eval('{}()'.format(generate_call))

        return model

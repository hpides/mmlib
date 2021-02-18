import abc
import os
import sys
import zipfile
from enum import Enum
from shutil import copyfile

import torch

from mmlib.persistence import AbstractPersistenceService
from util.helper import zip_dir, clean

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


class RecoverInfoT1(Enum):
    WEIGHTS = 'weights'
    MODEL_CODE = 'model_code'
    IMPORT_ROOT = 'import_root'
    CODE_NAME = 'code_name'
    RECOVER_VAL = 'recover_val'


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
    """TODO docs A Service that offers functionality to store PyTorch models. In order to do so it stores the metadata is
    stored in a MongoDB, the model (pickled) is stored on the file system. """

    def __init__(self, persistence_service: AbstractPersistenceService, tmp_path: str):
        """
        TODO docs
        :param base_path: The path that is used as a root directory for everything that is stored to the file system.
        :param host: The host name or Ip address to connect to a running MongoDB instance.
        """
        self._pers_service = persistence_service
        self._tmp_path = os.path.abspath(tmp_path)

    def save_model(self, model: torch.nn.Module, code: str, code_name: str, ) -> str:
        recover_info_t1 = self._save_model_t1(model, code, code_name)
        recover_info_id = self._pers_service.save_dict(recover_info_t1, RECOVER_T1)

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
        dst_path = os.path.join(self._tmp_path, self._pers_service.generate_id())

        zip_file = self._pickle_weights(model, dst_path)
        zip_file_id = self._pers_service.save_file(zip_file)
        code_file_id = self._pers_service.save_file(code)
        clean(dst_path)
        clean(zip_file)

        recover_info_t1 = {
            RecoverInfoT1.WEIGHTS.value: zip_file_id,
            RecoverInfoT1.MODEL_CODE.value: code_file_id,
            RecoverInfoT1.CODE_NAME.value: code_name,
            RecoverInfoT1.RECOVER_VAL.value: None  # TODO to implement
        }
        return recover_info_t1

    def save_version(self, model: torch.nn.Module, base_model_id: str) -> str:
        base_model_info = self._pers_service.recover_dict(base_model_id, MODEL_INFO)
        base_model_recover_info = self._get_recover_info(base_model_info)

        # copy fields from previous model that will stay the same
        code_name = base_model_recover_info[RecoverInfoT1.CODE_NAME.value]

        tmp_path = os.path.abspath(os.path.join(self._tmp_path, TMP_DIR))
        os.mkdir(tmp_path)  # TODO maybe use with context
        code = self._pers_service.recover_file(base_model_recover_info[RecoverInfoT1.MODEL_CODE.value], tmp_path)

        recover_info_t1 = self._save_model_t1(model, code, code_name)
        clean(tmp_path)

        recover_info_id = self._pers_service.save_dict(recover_info_t1, RECOVER_T1)

        # TODO to implement other fields that are default None
        model_id = self._save_model_info(SaveType.PICKLED_WEIGHTS.value, recover_info_id, derived_from=base_model_id)

        return model_id

    def _get_recover_info(self, base_model_info):
        recover_info_id = base_model_info[ModelInfo.RECOVER_INFO.value]
        base_model_recover_info = self._pers_service.recover_dict(recover_info_id, RECOVER_T1)
        return base_model_recover_info

    def saved_model_ids(self) -> [str]:
        pass
        # str_ids = list(map(str, self._mongo_service.get_ids()))
        # return str_ids

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
        recover_id = model_info[ModelInfo.RECOVER_INFO.value]
        recover_info = self._pers_service.recover_dict(recover_id, RECOVER_T1)
        weights_file_id = recover_info[RecoverInfoT1.WEIGHTS.value]

        tmp_path = os.path.abspath(os.path.join(self._tmp_path, TMP_DIR))
        os.mkdir(tmp_path)  # TODO maybe use with context
        code_id = recover_info[RecoverInfoT1.MODEL_CODE.value]
        code = self._pers_service.recover_file(code_id, tmp_path)
        generate_call = recover_info[RecoverInfoT1.CODE_NAME.value]
        model = self._init_model(code, generate_call)

        weights_file = self._pers_service.recover_file(weights_file_id, tmp_path)
        s_dict = self._recover_pickled_weights(weights_file, tmp_path)
        model.load_state_dict(s_dict)

        clean(tmp_path)

        return model

    def _recover_pickled_model(self, pickle_path, extract_path):

        unpacked_path = self._unzip(pickle_path, extract_path)
        # make available for imports
        sys.path.append(unpacked_path)

        pickle_path = os.path.join(unpacked_path, 'model')
        loaded_model = torch.load(pickle_path)
        return loaded_model

    def _recover_pickled_weights(self, weights_file, extract_path):
        unpacked_path = self._unzip(weights_file, extract_path)

        pickle_path = os.path.join(unpacked_path, 'model_weights')
        state_dict = torch.load(pickle_path)
        return state_dict

    def _unzip(self, file_path, extract_path):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        # remove .zip file ending
        unpacked_path = file_path.split('.')[0]

        return unpacked_path

    def _find_py_files(self, root):
        result = []
        for root, dirs, files in os.walk(root):
            for file in files:
                if file.endswith(".py"):
                    result.append(os.path.join(root, file))

        return result

    def _code_path(self, base_model_save_path):
        unzipped_root = self._unzip(base_model_save_path)
        # there should be only one file with ending ".py" and this should be the model
        py_files = self._find_py_files(unzipped_root)
        assert len(py_files) == 1
        code = py_files[0]
        return code

    def _pickle_model(self, model, code, import_root, save_path):
        # create directory to store in
        abs_save_path = os.path.abspath(save_path)
        os.makedirs(abs_save_path)

        # store pickle dump of model
        torch.save(model, os.path.join(abs_save_path, 'model'))

        # store code
        self._store_code(abs_save_path, code, import_root)

        # zip everything
        return self._zip(save_path)

    def _pickle_weights(self, model, save_path):
        # create directory to store in
        abs_save_path = os.path.abspath(save_path)
        os.makedirs(abs_save_path)

        # store pickle dump of model
        torch.save(model.state_dict(), os.path.join(abs_save_path, 'model_weights'))

        # zip everything
        return self._zip(save_path)

    def _store_code(self, abs_save_path, code, import_root):
        code_abs_path = os.path.abspath(code)
        import_root_abs = os.path.abspath(import_root)
        copy_path, code_file = os.path.split(os.path.relpath(code_abs_path, import_root_abs))
        net_code_dst = os.path.join(abs_save_path, copy_path)
        # create dir structure in tmp file, needed to restore the pickle dump
        os.makedirs(net_code_dst)
        copyfile(code_abs_path, os.path.join(net_code_dst, code_file))

    def _zip(self, save_path):
        path, name = os.path.split(save_path)
        # temporarily change dict for zip process
        owd = os.getcwd()
        os.chdir(path)
        zip_name = name + '.zip'
        zip_dir(name, zip_name)
        # change path back
        os.chdir(owd)

        return os.path.join(path, zip_name)

    def _add_save_path(self, model_id, save_path):
        attribute = {SAVE_PATH: save_path}
        self._mongo_service.add_attribute(model_id, attribute)

    def _init_model(self, code, generate_call):
        path, file = os.path.split(code)
        module = file.replace('.py', '')
        sys.path.append(path)
        exec('from {} import {}'.format(module, generate_call))
        model = eval('{}()'.format(generate_call))

        return model

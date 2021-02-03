import abc
import os
import sys
import zipfile
from enum import Enum
from shutil import copyfile

import bson
import torch

from mmlib.recover import FileSystemMongoRecoverService
from util.helper import zip_dir
from util.mongo import MongoService

SAVE_PATH = 'save-path'
SAVE_TYPE = 'save-type'
NAME = 'name'
MODELS = 'models'
MMLIB = 'mmlib'


class SaveType(Enum):
    PICKLED_MODEL = 1
    ARCHITECTURE_AND_WEIGHTS = 2
    PROVENANCE = 3


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
    def save_model(self, name: str, model: torch.nn.Module, code: str, import_root: str) -> str:
        """
        Saves a model as a pickle dump together with the given metadata.
        :param name: The name of the model as a string. Used for easier identification.
        :param model: The model object.
        :param code: The path to the code of the model (is needed for recover process)
        :param import_root: The directory that is root for all imports, e.g. the Python project root.
        :return: Returns the ID that was used to store the model data in the MongoDB.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save_version(self, model: torch.nn.Module, base_model_id: str) -> str:
        """
        Saves a new model version by referring to the base_model
        :param model: The model to save.
        :param base_model_id: the model id of the base_model
        :return: Returns the ID that was used to store the new model version data in the MongoDB.
        """

    @abc.abstractmethod
    def saved_model_ids(self) -> [str]:
        """Returns list of saved models ids"""
        raise NotImplementedError

    @abc.abstractmethod
    def model_save_size(self, model_id: str) -> float:
        """
        Calculates and returns the amount of bytes that are used for storing the model.
        :param model_id: The ID to identify the model in the mongoDB.
        :return: The amout of bytes used to store the model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def recover_model(self, model_id: str) -> torch.nn.Module:
        """
        Recovers a the model identified by the given id.
        :param model_id: The id to identify the model with.
        :return: The recovered model as an object.
        """


class FileSystemMongoSaveService(AbstractSaveRecoverService):
    """A Service that offers functionality to store PyTorch models. In order to do so it stores the metadata is
    stored in a MongoDB, the model (pickled) is stored on the file system. """

    def __init__(self, base_path, host='127.0.0.1'):
        """
        :param base_path: The path that is used as a root directory for everything that is stored to the file system.
        :param host: The host name or Ip address to connect to a running MongoDB instance.
        """
        self._mongo_service = MongoService(host, MMLIB, MODELS)
        self._base_path = base_path
        self._recover_service = FileSystemMongoRecoverService(base_path, host)

    def save_model(self, name: str, model: torch.nn.Module, code: str, import_root: str) -> str:
        model_dict = {
            NAME: name,
            SAVE_TYPE: SaveType.PICKLED_MODEL.value
        }

        model_id = self._mongo_service.save_dict(model_dict)

        save_path = os.path.join(self._base_path, str(model_id) + '.zip')
        attribute = {SAVE_PATH: save_path}

        self._pickle_model(model, code, import_root, os.path.join(self._base_path, str(model_id)))

        self._mongo_service.add_attribute(model_id, attribute)

        return str(model_id)

    def save_version(self, model: torch.nn.Module, base_model_id: str) -> str:
        pass

    def _pickle_model(self, model, code, import_root, save_path):
        # create directory to store in
        abs_save_path = os.path.abspath(save_path)
        os.makedirs(abs_save_path)

        # store pickle dump of model
        torch.save(model, os.path.join(abs_save_path, 'model'))

        # store code
        code_abs_path = os.path.abspath(code)
        import_root_abs = os.path.abspath(import_root)
        copy_path, code_file = os.path.split(os.path.relpath(code_abs_path, import_root_abs))
        net_code_dst = os.path.join(abs_save_path, copy_path)

        # create dir structure in tmp file, needed to restore the pickle dump
        os.makedirs(net_code_dst)
        copyfile(code_abs_path, os.path.join(net_code_dst, code_file))

        # zip everything
        path, name = os.path.split(save_path)
        # temporarily change dict for zip process
        owd = os.getcwd()
        os.chdir(path)
        zip_dir(name, name + '.zip')

        # change path back
        os.chdir(owd)

    def saved_model_ids(self) -> [str]:
        str_ids = list(map(str, self._mongo_service.get_ids()))
        return str_ids

    def model_save_size(self, model_id: str) -> float:
        model_id = bson.ObjectId(model_id)

        document_size = self._mongo_service.document_size(model_id)

        meta_data = self._mongo_service.get_dict(model_id)
        save_path = meta_data[SAVE_PATH]
        zip_size = os.path.getsize(save_path)

        return document_size + zip_size

    def recover_model(self, model_id: str) -> torch.nn.Module:
        model_id = bson.ObjectId(model_id)
        model_dict = self._mongo_service.get_dict(model_id)
        return self._recover_model(model_dict)

    def _recover_model(self, model_dict):
        save_type = SaveType(model_dict[SAVE_TYPE])
        if save_type == SaveType.PICKLED_MODEL:
            return self._restore_pickled_model(model_dict)

    def _restore_pickled_model(self, model_dict):
        file_path = model_dict[SAVE_PATH]

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(self._base_path)

        # remove .zip file ending
        unpacked_path = file_path.split('.')[0]
        # make available for imports
        sys.path.append(unpacked_path)

        pickle_path = os.path.join(unpacked_path, 'model')
        loaded_model = torch.load(pickle_path)
        return loaded_model

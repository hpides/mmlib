import abc
import os
import sys
import zipfile

import bson
import torch

from mmlib.save import MMLIB, MODELS, SAVE_TYPE, SaveType, SAVE_PATH
from util.mongo import MongoService


class RecoverService(metaclass=abc.ABCMeta):
    """A Service that offers functionality to recover PyTorch models from given data."""

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'recover_model') and
                callable(subclass.recover_model) or
                NotImplemented)

    @abc.abstractmethod
    def recover_model(self, model_id: bson.ObjectId) -> torch.nn.Module:
        """
        Recovers a the model identified by the given id.
        :param model_id: The id to identify the model with.
        :return: The recovered model as an object.
        """


class FileSystemMongoRecoverService(RecoverService):
    """A Service that offers functionality to recover PyTorch models that have been stored using the
    FileSystemMongoSaveService. """

    def __init__(self, base_path, host='127.0.0.1'):
        """
        :param base_path: The path that is used as a root directory for everything that is stored to the file system.
        :param host: The host name or Ip address to connect to a running MongoDB instance.
        """
        self._mongo_service = MongoService(host, MMLIB, MODELS)
        self._base_path = base_path

    def recover_model(self, model_id: bson.ObjectId) -> torch.nn.Module:
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

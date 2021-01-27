import os
import sys
import zipfile
from enum import Enum
from shutil import copyfile

import torch

from util.mongo import MongoService
from util.helper import zip_dir

SAVE_PATH = 'save-path'
SAVE_TYPE = 'save-type'
NAME = 'name'
MODELS = 'models'
MMLIB = 'mmlib'


class SaveType(Enum):
    PICKLED_MODEL = 1
    ARCHITECTURE_AND_WEIGHTS = 2
    PROVENANCE = 3


class SaveService:
    """A Service that offers functionality to store PyTorch models."""

    def __init__(self, base_path, host='127.0.0.1'):
        """
        :param base_path: The path that is used as a root directory for everything that is stored to the file system.
        :param host: The host name or Ip address to connect to a running MongoDB instance.
        """
        self._mongo_service = MongoService(host, MMLIB, MODELS)
        self._base_path = base_path

    def save_model(self, name, model, code, import_root):
        """
        Saves a model as a pickle dump together with the given metadata.
        The metadata is stored in a MongoDB, the model (pickled) is stored on the file system.
        :param name: The name of the model as a string. Used for easier identification.
        :param model: The model object.
        :param code: The path to the code of the model (is needed for recover process)
        :param import_root: The directory that is root for all imports, e.g. the Python project root.
        :return: Returns the ID that was sued to store the model data in the MongoDB.
        """
        model_dict = {
            NAME: name,
            SAVE_TYPE: SaveType.PICKLED_MODEL.value
        }

        model_id = self._mongo_service.save_dict(model_dict)

        save_path = os.path.join(self._base_path, str(model_id) + '.zip')
        attribute = {SAVE_PATH: save_path}

        self._pickle_model(model, code, import_root, os.path.join(self._base_path, str(model_id)))

        self._mongo_service.add_attribute(model_id, attribute)

        return model_id

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

    # def save_model(self, name, architecture, model):
    #     pass
    #
    # def save_model(self, name, provenance):
    #     pass

    def saved_model_ids(self):
        """Returns list of saved models ids"""
        return self._mongo_service.get_ids()


class RecoverService:
    """A Service that offers functionality to recover PyTorch models from given data."""

    def __init__(self, base_path, host='127.0.0.1'):
        """
        :param base_path: The path that is used as a root directory for everything that is stored to the file system.
        :param host: The host name or Ip address to connect to a running MongoDB instance.
        """
        self._mongo_service = MongoService(host, MMLIB, MODELS)
        self._base_path = base_path

    def recover_model(self, model_id):
        """
        Recovers a the model identified by the given id.
        :param model_id: The id t identify the model with.
        :return: The recovered model as an object.
        """
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

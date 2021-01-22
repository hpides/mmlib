import os
from enum import Enum
from shutil import copyfile

import torch

from mmlib.mongo import MongoService
from mmlib.util import zip_dir

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
    def __init__(self, base_path, host='127.0.0.1'):
        self._mongo_service = MongoService(host, MMLIB, MODELS)
        self._base_path = base_path

    def save_model(self, name, model, code, import_root, dst):
        model_dict = {
            NAME: name,
            SAVE_TYPE: SaveType.PICKLED_MODEL.value
        }

        model_id = self._mongo_service.save_dict(model_dict)

        save_path = os.path.join(self._base_path, str(model_id))
        attribute = {SAVE_PATH: save_path}

        self._pickle_model(model, code, import_root, os.path.join(dst, str(model_id)))

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
        copyfile(code, os.path.join(net_code_dst, code_file))

        path, name = os.path.split(save_path)
        os.chdir(path)
        zip_dir(name, name + '.zip')

    # def save_model(self, name, architecture, model):
    #     pass
    #
    # def save_model(self, name, provenance):
    #     pass

    def saved_model_ids(self):
        """Returns list of saved models ids"""
        return self._mongo_service.get_ids()

    def recover_model(self, model_id):
        model_dict = self._mongo_service.get_dict(model_id)
        return self._recover_model(model_dict)

    def _recover_model(self, model_dict):
        save_type = SaveType(model_dict[SAVE_TYPE])
        if save_type == SaveType.PICKLED_MODEL:
            return self._restore_pickled_model(model_dict)

    def _restore_pickled_model(self, model_dict):
        # TODO think about warning
        # TODO check wht restrictions we have with pickled models
        file_path = model_dict[SAVE_PATH]
        loaded = torch.load(file_path)
        return loaded

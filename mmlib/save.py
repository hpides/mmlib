import os
from enum import Enum

import torch

from mmlib.mongo import MongoService

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

    def save_model(self, name, model):
        model_dict = {
            NAME: name,
            SAVE_TYPE: SaveType.PICKLED_MODEL.value
        }

        model_id = self._mongo_service.save_dict(model_dict)

        save_path = os.path.join(self._base_path, str(model_id))
        attribute = {SAVE_PATH: save_path}

        torch.save(model, save_path)

        self._mongo_service.add_attribute(model_id, attribute)

        return model_id

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

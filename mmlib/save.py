import os
from enum import Enum

import torch
from pymongo import MongoClient

SAVE_PATH = 'save-path'
STORE_TYPE = 'store-type'
NAME = 'name'


class SaveType(Enum):
    PICKLED_MODEL = 1
    ARCHITECTURE_AND_WEIGHTS = 2
    PROVENANCE = 3


class SaveService:
    def __init__(self, base_path, host='127.0.0.1'):
        self._mongo_service = MongoService(host)
        self._base_path = base_path

    def save_model(self, name, model):
        model_dict = {
            NAME: name,
            STORE_TYPE: SaveType.PICKLED_MODEL.value
        }

        model_id = self._mongo_service.save_dict(model_dict)

        save_path = os.path.join(self._base_path, str(model_id))
        attribute = {SAVE_PATH: save_path}

        torch.save(model, save_path)

        self._mongo_service.add_attribute(model_id, attribute)

        return model_id

    def saved_models(self):
        pass


MODELS = 'models'
MMLIB = 'mmlib'
ID = '_id'
SET = "$set"


class MongoService(object):
    def __init__(self, host):
        self._mongo_client = MongoClient(host)
        self._db_name = MMLIB
        self._collection_name = MODELS
        # close connection for now, for new requests there will be a reconnect
        self._mongo_client.close()

    def save_dict(self, insert_dict):
        """
        Saves a python dictionary in the underlying mongoDB.
        :param insert_dict: The dictionary to save.
        :return: The id that was automatically assigned by mongoDB.
        """
        collection = self._get_collection()
        feedback = collection.insert_one(insert_dict)

        return feedback.inserted_id

    def get_ids(self):
        """
        Gets all ids that can be found in the used mongoDB collection
        :return: The list of ids.
        """
        collection = self._get_collection()

        return collection.find({}).distinct(ID)

    def get_model_dict(self, model_id):
        """
        Retrieves the model dict identified by its mongoDB id.
        :param model_id: The mongoDB id to find the model dict.
        :return: The retrieved model dict.
        """
        collection = self._get_collection()

        return collection.find({ID: model_id})[0]

    def add_attribute(self, model_id, attribute):
        """
        Adds an attribute to an entry identified by the model_id.
        :param model_id: The model_id to identify the entry to modify.
        :param attribute: The attribute(s) to add.
        """
        collection = self._get_collection()

        query = {ID: model_id}
        new_values = {SET: attribute}

        collection.update_one(query, new_values)

    def _get_collection(self):
        db = self._mongo_client[self._db_name]
        collection = db[self._collection_name]
        return collection

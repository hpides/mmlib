import bson
from bson import ObjectId
from pymongo import MongoClient

from mmlib.constants import ID

STOP = 'STOP'

START = 'START'

_ID = '_id'
SET = "$set"


class MongoService(object):
    def __init__(self, host, db_name, logging=False):
        self._mongo_client = MongoClient(host)
        self._db_name = db_name
        self.logging = logging
        # close connection for now, for new requests there will be a reconnect
        self._mongo_client.close()

    def save_dict(self, insert_dict: dict, collection: str, id: str = None) -> ObjectId:
        """
        Saves a python dictionary in the underlying mongoDB.
        :param insert_dict: The dictionary to save.
        :param collection: The mongo collection to use.
        :param id: Optional id to use, otherwise id will be auto generated
        :return: The id that was automatically assigned by mongoDB.
        """
        collection = self._get_collection(collection)

        # if id filed is already set then use it as mongoID
        if ID in insert_dict:
            obj_id = ObjectId(insert_dict[ID])
            insert_dict[_ID] = obj_id
        # if id is given as parameter use it as mongoID
        elif id:
            obj_id = ObjectId(id)
            insert_dict[_ID] = obj_id

        feedback = collection.insert_one(insert_dict)

        if id:
            assert str(feedback.inserted_id) == id, 'given id was not used'

        return feedback.inserted_id

    def get_ids(self, collection: str) -> [ObjectId]:
        """
        Gets all ids that can be found in the used mongoDB collection
        :param collection: The mongo collection to use.
        :return: The list of ids.
        """
        collection = self._get_collection(collection)

        return collection.find({}).distinct(_ID)

    def get_dict(self, object_id: ObjectId, collection: str) -> dict:
        """
        Retrieves the dict identified by its mongoDB id.
        :param object_id: The mongoDB id to find the dict.
        :param collection: The mongo collection to use.
        :return: The retrieved dict.
        """
        collection = self._get_collection(collection)

        return collection.find({_ID: object_id})[0]

    def add_attribute(self, object_id: ObjectId, attribute, collection: str):
        """
        Adds an attribute to an entry identified by the object_id.
        :param object_id: The id to identify the entry to modify.
        :param attribute: The attribute(s) to add.
        :param collection: The mongo collection to use.
        """
        collection = self._get_collection(collection)

        query = {_ID: object_id}
        new_values = {SET: attribute}

        collection.update_one(query, new_values)

    def document_size(self, object_id: ObjectId, collection: str) -> int:
        """
        Calculated the size in bytes of a document identified by its object_id.
        :param object_id: The id to identify the document.
        :param collection: The mongo collection to use.
        :return: The document size in bytes.
        """
        collection = self._get_collection(collection)
        item = collection.find({_ID: object_id})[0]
        return len(bson.BSON.encode(item))

    def id_exists(self, object_id: ObjectId, collection: str) -> bool:
        collection = self._get_collection(collection)

        return collection.find({'$exists': {_ID: object_id}})

    def _get_collection(self, collection_name):
        collection_name = collection_name.lower()
        db = self._mongo_client[self._db_name]
        collection = db[collection_name]
        return collection

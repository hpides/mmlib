from pymongo import MongoClient

ID = '_id'
SET = "$set"


class MongoService(object):
    def __init__(self, host, db_name, collection_name):
        self._mongo_client = MongoClient(host)
        self._db_name = db_name
        self._collection_name = collection_name
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

    def get_dict(self, object_id):
        """
        Retrieves the dict identified by its mongoDB id.
        :param object_id: The mongoDB id to find the dict.
        :return: The retrieved dict.
        """
        collection = self._get_collection()

        return collection.find({ID: object_id})[0]

    def add_attribute(self, object_id, attribute):
        """
        Adds an attribute to an entry identified by the object_id.
        :param object_id: The id to identify the entry to modify.
        :param attribute: The attribute(s) to add.
        """
        collection = self._get_collection()

        query = {ID: object_id}
        new_values = {SET: attribute}

        collection.update_one(query, new_values)

    def _get_collection(self):
        db = self._mongo_client[self._db_name]
        collection = db[self._collection_name]
        return collection

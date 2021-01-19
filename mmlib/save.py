from pymongo import MongoClient


class SaveService:
    def __init__(self, host='127.0.0.1'):
        self.mongo_service = MongoService(host)

    def model_to_dict(self, model):
        pass

    def save_model(self, model):
        model_dict = self.model_to_dict(model)
        id = self.mongo_service.save_dict(model_dict)
        return id

    def saved_models(self):
        pass


MODELS = 'models'
MMLIB = 'mmlib'
ID = '_id'


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

    def _get_collection(self):
        db = self._mongo_client[self._db_name]
        collection = db[self._collection_name]
        return collection

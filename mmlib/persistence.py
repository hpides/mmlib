import abc
import os
from shutil import copyfile

from bson import ObjectId

from util.helper import find_file
from util.mongo import MongoService

MMLIB = 'mmlib'
FILE = 'file-'
DICT = 'dict-'


class AbstractPersistenceService(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def save_dict(self, insert_dict: dict, represent_type: str) -> str:
        """
        Persists a python dictionary.
        :param insert_dict: The dict that should be persisted.
        :param represent_type: The type of the dict to store.
        :return: The id that was used to store the dictionary.
        """

    @abc.abstractmethod
    def recover_dict(self, dict_id: str, represent_type: str) -> dict:
        """
        Recovers a dictionary.
        :param dict_id: The id that identifies the dictionary to recover.
        :param represent_type: The type of the dict to recover.
        :return: The recovered python dictionary.
        """

    @abc.abstractmethod
    def save_file(self, file_path: str) -> str:
        """
        Persists a file.
        :param file_path: The path the file to persist.
        :return: The id that was used to store the file.
        """

    @abc.abstractmethod
    def recover_file(self, file_id: str, dst_path) -> str:
        """
        Recovers a file.
        :param file_id: The id that identifies the file to recover.
        :param dst_path: The path where the restored file should be stored to.
        :return: The path to the restored file.
        """

    @abc.abstractmethod
    def generate_id(self) -> str:
        """
        Generates an id as a string.
        :return: The generated id.
        """

    @abc.abstractmethod
    def get_all_dict_ids(self, represent_type: str) -> [str]:
        """
        Returns all ids for a given type.
        :param represent_type: The type of the collection to get the ids for.
        :return: all ids as a list of strings
        """


class FileSystemMongoPS(AbstractPersistenceService):

    def __init__(self, base_path, host='127.0.0.1'):
        self._mongo_service = MongoService(host, MMLIB)
        self._base_path = os.path.abspath(base_path)

    def save_dict(self, insert_dict: dict, represent_type: str) -> str:
        mongo_id = self._mongo_service.save_dict(insert_dict, collection=represent_type)
        return DICT + str(mongo_id)

    def recover_dict(self, dict_id: str, represent_type: str) -> dict:
        mongo_dict_id = ObjectId(dict_id.replace(DICT, ''))
        return self._mongo_service.get_dict(mongo_dict_id, collection=represent_type)

    def save_file(self, file_path: str) -> str:
        path, file_name = os.path.split(file_path)
        file_id = str(ObjectId())
        dst_path = os.path.join(self._base_path, file_id)
        os.mkdir(dst_path)
        copyfile(file_path, os.path.join(dst_path, file_name))

        return FILE + file_id

    def recover_file(self, file_id: str, dst_path):
        file_id = file_id.replace(FILE, '')
        store_path = os.path.join(self._base_path, file_id)
        file = find_file(store_path)
        dst = os.path.join(os.path.abspath(dst_path), os.path.split(file)[1])
        copyfile(file, dst)

        return dst

    def generate_id(self) -> str:
        return str(ObjectId())

    def get_all_dict_ids(self, represent_type: str) -> [str]:
        mongo_ids = self._mongo_service.get_ids(represent_type)
        return list(map(lambda i: '{}{}'.format(DICT, str(i)), mongo_ids))

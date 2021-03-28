import abc
import os
from shutil import copyfile

from bson import ObjectId

from util.helper import find_file
from util.mongo import MongoService


class AbstractPersistenceService(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def generate_id(self) -> str:
        """
        Generates an id as a string.
        :return: The generated id.
        """

    @abc.abstractmethod
    def get_all_ids(self) -> [str]:
        """
        Returns all ids that are in use.
        :return: ids as a list of strings.
        """

    @abc.abstractmethod
    def is_valid_id(self, pers_id: str) -> bool:
        """
        Checks if the given id is a valid id for the PersistenceService.
        :param pers_id: The type of the collection to get the ids for.
        :return: True if the pers_id is valid.
        """


class AbstractDictPersistenceService(AbstractPersistenceService, metaclass=abc.ABCMeta):

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
    def dict_size(self, dict_id: str, represent_type: str) -> int:
        """
        Calculates and returns the size of a dict in bytes.
        :param dict_id: The id to identify the dict.
        :param represent_type: The type of the collection to get the ids for.
        :return: The dict size in bytes.
        """

    @abc.abstractmethod
    def id_exists(self, dict_id: str, represent_type: str) -> bool:
        """
        Checks if the given id exists already
        :param dict_id: the id to check for
        :param represent_type: The type of the collection to get the ids for.
        :return: true if the id already exists, false otherwise
        """

    @abc.abstractmethod
    def all_ids_for_type(self, represent_type: str) -> [str]:
        """
        Returns all ids for a given type.
        :param represent_type: The type of the collection to get the ids for.
        :return: all ids as a list of strings
        """


class AbstractFilePersistenceService(AbstractPersistenceService, metaclass=abc.ABCMeta):

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
    def file_size(self, file_id: str) -> int:
        """
        Calculates and returns the size of a file in bytes.
        :param file_id: The id to identify the file.
        :return: The file size in bytes.
        """


FILE = 'file-'
MMLIB = 'mmlib'


class FileSystemPersistenceService(AbstractFilePersistenceService):

    def __init__(self, base_path):
        self._base_path = os.path.abspath(base_path)

    def save_file(self, file_path: str) -> str:
        path, file_name = os.path.split(file_path)
        file_id = str(ObjectId())
        dst_path = self._get_store_path(file_id)
        os.mkdir(dst_path)
        copyfile(file_path, os.path.join(dst_path, file_name))

        return FILE + file_id

    def recover_file(self, file_id: str, dst_path):
        file_id = self._to_internal_file_id(file_id)
        store_path = self._get_store_path(file_id)
        file = find_file(store_path)
        dst = os.path.join(os.path.abspath(dst_path), os.path.split(file)[1])
        copyfile(file, dst)

        return dst

    def file_size(self, file_id: str) -> int:
        file_id = self._to_internal_file_id(file_id)
        store_path = self._get_store_path(file_id)
        file = find_file(store_path)

        return os.path.getsize(file)

    def generate_id(self) -> str:
        return str(ObjectId())

    def is_file_ref(self, field: str) -> bool:
        return field.startswith(FILE)

    def get_all_ids(self) -> [str]:
        # TODO
        pass

    def is_valid_id(self, pers_id: str) -> bool:
        # TODO
        pass

    def _to_internal_file_id(self, file_id):
        return file_id.replace(FILE, '')

    def _get_store_path(self, file_id):
        store_path = os.path.join(self._base_path, file_id)
        return store_path


DICT = 'dict-'


class MongoDictPersistenceService(AbstractDictPersistenceService):

    def __init__(self, host='127.0.0.1'):
        self._mongo_service = MongoService(host, MMLIB)

    def generate_id(self) -> str:
        return str(ObjectId())

    def save_dict(self, insert_dict: dict, represent_type: str) -> str:
        mongo_id = self._mongo_service.save_dict(insert_dict, collection=represent_type)
        return str(mongo_id)

    def recover_dict(self, dict_id: str, represent_type: str) -> dict:
        mongo_dict_id = self._to_mongo_dict_id(dict_id)
        return self._mongo_service.get_dict(mongo_dict_id, collection=represent_type)

    def all_ids_for_type(self, represent_type: str) -> [str]:
        mongo_ids = self._mongo_service.get_ids(represent_type)
        return ['{}{}'.format(DICT, str(i)) for i in mongo_ids]

    def dict_size(self, dict_id: str, represent_type: str) -> int:
        dict_id = self._to_mongo_dict_id(dict_id)
        return self._mongo_service.document_size(dict_id, represent_type)

    def is_dict_ref(self, field: str) -> bool:
        return field.startswith(DICT)

    def id_exists(self, dict_id: str, represent_type: str) -> bool:
        dict_id = self._to_mongo_dict_id(dict_id)
        return self._mongo_service.id_exists(dict_id, represent_type)


    def get_all_ids(self) -> [str]:
        # TODO
        pass

    def is_valid_id(self, pers_id: str) -> bool:
        # TODO
        pass

    def _to_mongo_dict_id(self, dict_id):
        return ObjectId(dict_id.replace(DICT, ''))

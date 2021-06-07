import abc
import json
import os
import time
import uuid
from shutil import copyfile

from bson import ObjectId

from schema.file_reference import FileReference
from util.helper import find_file, log_stop, START_STOP, TIME
from util.mongo import MongoService

MMLIB_FILE_PERS = 'mmlib_file_pers'

MMLIB_DICT_PERS = 'mmlib_dict_pers'

START = 'start'


class PersistenceService(metaclass=abc.ABCMeta):

    def __init__(self, logging=False):
        self.logging = logging

    @abc.abstractmethod
    def generate_id(self) -> str:
        """
        Generates an id as a string.
        :return: The generated id.
        """


class DictPersistenceService(PersistenceService, metaclass=abc.ABCMeta):

    def __init__(self, logging=False):
        super().__init__(logging)

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

    @abc.abstractmethod
    def add_field(self, dict_id: str, represent_type: str, add_dict: dict):
        """
        Adds a field to a given, stored dictionary
        :param dict_id: The id of the dictionary to add the field to.
        :param represent_type: The type of the collection.
        :param add_dict: The data that should be added represented as a dict
        """


class FilePersistenceService(PersistenceService, metaclass=abc.ABCMeta):

    def __init__(self, logging=False):
        super().__init__(logging)

    @abc.abstractmethod
    def save_file(self, file: FileReference):
        """
        Persists the file given by the file reference, the id that was used to store the file is set for the given
        file reference.
        :param file: The file to persist given as a FileReference.
        """

    @abc.abstractmethod
    def recover_file(self, file: FileReference, dst_path):
        """
        Recovers the file given by the file reference, the path for the restored file is set for the given
        file reference.
        :param file: The file to recover identified by FileReference.
        :param dst_path: The path where the restored file should be stored to.
        """

    @abc.abstractmethod
    def file_size(self, file: FileReference):
        """
        Calculates and sets the size property for the given file in bytes.
        :param file: The file identified by FileReference.
        """


FILE = 'file-'
MMLIB = 'mmlib'


class FileSystemPersistenceService(FilePersistenceService):

    def __init__(self, base_path, logging=False):
        super().__init__(logging)
        assert os.path.isdir(base_path), 'base_path ({}) does not seem to exist'.format(base_path)
        self._base_path = os.path.abspath(base_path)

    def save_file(self, file: FileReference):
        log = self.log_start('save_file')
        path, file_name = os.path.split(file.path)
        file_id = str(ObjectId())
        dst_path = self._get_store_path(file_id)
        os.mkdir(dst_path)
        copyfile(file.path, os.path.join(dst_path, file_name))
        file.reference_id = FILE + file_id
        self.log_stop(log)

    def recover_file(self, file: FileReference, dst_path):
        log = self.log_start('recover_file')
        internal_file_id = self._to_internal_file_id(file.reference_id)
        store_path = self._get_store_path(internal_file_id)
        file_path = find_file(store_path)
        dst = os.path.join(os.path.abspath(dst_path), os.path.split(file_path)[1])

        assert not os.path.isfile(dst), 'file at {} exists already'.format(dst)
        copyfile(file_path, dst)
        file.path = dst
        self.log_stop(log)

    def file_size(self, file: FileReference):
        internal_file_id = self._to_internal_file_id(file.reference_id)
        store_path = self._get_store_path(internal_file_id)
        file_path = find_file(store_path)
        file.size = os.path.getsize(file_path)

    def generate_id(self) -> str:
        return str(uuid.uuid4())

    def is_file_ref(self, field: str) -> bool:
        return field.startswith(FILE)

    def _to_internal_file_id(self, file_id):
        return file_id.replace(FILE, '')

    def _get_store_path(self, file_id):
        store_path = os.path.join(self._base_path, file_id)
        return store_path

    def log_start(self, method):
        if self.logging:
            t = time.time_ns()
            _id = uuid.uuid4()
            log_dict = {
                START_STOP: START,
                '_id': str(_id),
                'service': MMLIB_FILE_PERS,
                'method': method,
                TIME: t
            }

            print(json.dumps(log_dict))

            return log_dict

    def log_stop(self, log_dict):
        log_stop(self.logging, log_dict)


DICT = 'dict-'


class MongoDictPersistenceService(DictPersistenceService):

    def __init__(self, logging=False, host='127.0.0.1'):
        super().__init__(logging)
        self._mongo_service = MongoService(host, MMLIB)

    def generate_id(self) -> str:
        return str(ObjectId())

    def save_dict(self, insert_dict: dict, represent_type: str) -> str:
        log = self.log_start('save_dict', represent_type)
        mongo_id = self._mongo_service.save_dict(insert_dict, collection=represent_type)
        self.log_stop(log)
        return str(mongo_id)

    def recover_dict(self, dict_id: str, represent_type: str) -> dict:
        log = self.log_start('recover_dict', represent_type)
        mongo_dict_id = self._to_mongo_dict_id(dict_id)
        self.log_stop(log)
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

    def add_field(self, dict_id: str, represent_type: str, add_dict: dict):
        log = self.log_start('add_field', represent_type)
        dict_id = self._to_mongo_dict_id(dict_id)
        self._mongo_service.add_attribute(dict_id, add_dict, represent_type)
        self.log_stop(log)

    def _to_mongo_dict_id(self, dict_id):
        return ObjectId(dict_id.replace(DICT, ''))

    def log_start(self, method, collection):
        if self.logging:
            t = time.time_ns()
            _id = uuid.uuid4()
            log_dict = {
                START_STOP: START,
                '_id': str(_id),
                'service': MMLIB_DICT_PERS,
                'method': method,
                'collection': collection,
                TIME: t
            }

            print(json.dumps(log_dict))

            return log_dict

    def log_stop(self, log_dict):
        log_stop(self.logging, log_dict)

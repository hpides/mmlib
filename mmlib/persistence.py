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
        TODO docs
        :param insert_dict:
        :param represent_type:
        :return:
        """

    @abc.abstractmethod
    def recover_dict(self, dict_id: str, represent_type: str) -> dict:
        """
        TODO docs
        :param dict_id:
        :param represent_type:
        :return:
        """

    @abc.abstractmethod
    def save_file(self, file_path: str) -> str:
        """
        TODO docs
        :param file_path:
        :return:
        """

    @abc.abstractmethod
    def recover_file(self, file_id: str, dst_path):
        """
        TODO docs
        :param file_id:
        :param dst_path:
        :return:
        """

    @abc.abstractmethod
    def generate_id(self) -> str:
        """
        TODO docs
        :return:
        """


# TODO move in separate file
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

    def generate_id(self) -> str:
        return str(ObjectId())

import abc
import os
from shutil import copyfile

from bson import ObjectId

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
    def save_file(self, file_path: str) -> str:
        """
        TODO docs
        :param file_path:
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
        self._base_path = base_path

    def save_dict(self, insert_dict: dict, represent_type: str) -> str:
        mongo_id = self._mongo_service.save_dict(insert_dict, collection=represent_type)
        return DICT + str(mongo_id)

    def save_file(self, file_path: str) -> str:
        path, file_name = os.path.split(file_path)
        file_id = str(ObjectId())
        dst_path = os.path.join(self._base_path, file_id)
        os.mkdir(dst_path)
        copyfile(file_path, os.path.join(dst_path, file_name))

        return FILE + file_id

    def generate_id(self) -> str:
        return str(ObjectId())

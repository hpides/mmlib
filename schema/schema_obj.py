import abc
import json

from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService


class SchemaObj(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def persist(self, file_pers_service: AbstractFilePersistenceService,
                dict_pers_service: AbstractDictPersistenceService) -> str:
        """
        Persists the schema object.
        :param file_pers_service: An instance of AbstractFilePersistenceService that is used to store files.
        :param dict_pers_service: An instance of AbstractDictPersistenceService that is used to store metadata as dicts.
        """

    @classmethod
    @abc.abstractmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService, restore_root: str):
        """
        Loads the schema object from database/disk.
        :param obj_id: The identifier for the SchemaObj in the database/disk.
        :param file_pers_service: An instance of AbstractFilePersistenceService that is used to store files.
        :param dict_pers_service: An instance of AbstractDictPersistenceService that is used to store metadata as dicts.
        :param restore_root: The path where restored files are stored to.
        """
        pass

    @abc.abstractmethod
    def size_in_bytes(self, file_pers_service: AbstractFilePersistenceService,
                      dict_pers_service: AbstractDictPersistenceService) -> int:
        """
        Calculates and returns the size of the SchemaObj in bytes.
        :param file_pers_service: An instance of AbstractFilePersistenceService that is used to store and load files.
        :param dict_pers_service: An instance of AbstractDictPersistenceService that is used to store and load metadata
         as dicts.
        :return: The size in bytes.
        """
        pass


def __eq__(self, other):
    self_dict = self.to_dict()
    other_dict = other.to_dict()

    return self_dict == other_dict


def __hash__(self):
    return hash(json.dumps(self.to_dict(), sort_keys=True))

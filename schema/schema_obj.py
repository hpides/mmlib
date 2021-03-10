import abc
from enum import Enum

from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService


class SchemaObjType(Enum):
    STRING = 'string'
    FILE = 'file'
    MODEL_INFO = 'ModelInfo'
    RECOVER_T1 = 'RecoverInfoT1'
    RECOVER_VAL = 'RecoverVal'
    DATASET = 'Dataset'
    TRAIN_INFO = 'TrainInfo'
    RESTORABLE_OBJ = 'restorable_obj'
    ENVIRONMENT = 'environment'
    FUNCTION = 'function'


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

# @abc.abstractmethod
# def to_dict(self) -> dict:
#     """
#     Represents a dict representation of the SchemaObj
#     :return: The dict representation
#     """
#
# @abc.abstractmethod
# def load_dict(self, state_dict: dict):
#     """
#     Update the internal state based on the given dict.
#     :param state_dict: The dict to load the state from.
#     """
#
# @abc.abstractmethod
# def get_type(self, dict_key) -> SchemaObjType:
#     """
#     Maps a dict key to the type that is used to store it.
#     :param dict_key: The key of the dict to request the type for.
#     :return: The type as an objet of SchemaObjType.
#     """
#
# def __eq__(self, other):
#     self_dict = self.to_dict()
#     other_dict = other.to_dict()
#
#     return self_dict == other_dict
#
# def __hash__(self):
#     return hash(json.dumps(self.to_dict(), sort_keys=True))

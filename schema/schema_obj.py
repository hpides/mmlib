import abc
import json
from enum import Enum


class SchemaObjType(Enum):
    STRING = 'string'
    FILE = 'file'
    MODEL_INFO = 'ModelInfo'
    RECOVER_T1 = 'RecoverInfoT1'
    RECOVER_Val = 'RecoverVal'


class SchemaObj(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def to_dict(self) -> dict:
        """
        Represents a dict representation of the SchemaObj
        :return: The dict representation
        """

    @abc.abstractmethod
    def load_dict(self, state_dict: dict):
        """
        Update the internal state based on the given dict.
        :param state_dict: The dict to load the state from.
        """

    @abc.abstractmethod
    def get_type(self, dict_key) -> SchemaObjType:
        """
        Maps a dict key to the type that is used to store it.
        :param dict_key: The key of the dict to request the type for.
        :return: The type as an objet of SchemaObjType.
        """

    def __eq__(self, other):
        self_dict = self.to_dict()
        other_dict = other.to_dict()

        return self_dict == other_dict

    def __hash__(self):
        return hash(json.dumps(self.to_dict(), sort_keys=True))

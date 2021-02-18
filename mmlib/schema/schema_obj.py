import abc
import json
from enum import Enum


class SchemaObjType(Enum):
    MODEL_INFO = 'model_info'
    RECOVER_T1 = 'recover_t1'


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
        Updtae the internal state based on the given dict.
        :param state_dict: The dict to load the state from.
        """

    def __eq__(self, other):
        self_dict = self.to_dict()
        other_dict = other.to_dict()

        return self_dict == other_dict

    def __hash__(self):
        return hash(json.dumps(self.to_dict(), sort_keys=True))

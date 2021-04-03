import abc

from mmlib.constants import ID
from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService


class SchemaObj(metaclass=abc.ABCMeta):

    def __init__(self, store_id: str = None):
        self.store_id = store_id

    def persist(self, file_pers_service: AbstractFilePersistenceService,
                dict_pers_service: AbstractDictPersistenceService) -> str:
        """
        Persists the schema object.
        :param file_pers_service: An instance of AbstractFilePersistenceService that is used to store files.
        :param dict_pers_service: An instance of AbstractDictPersistenceService that is used to store metadata as dicts.
        """
        if self.store_id and dict_pers_service.id_exists(self.store_id, self._representation_type()):
            # if the id already exists, we do not have to persist again
            return self.store_id

        if not self.store_id:
            self.store_id = dict_pers_service.generate_id()

        dict_representation = {
            ID: self.store_id,
        }

        self._persist_class_specific_fields(dict_representation, file_pers_service, dict_pers_service)

        dict_pers_service.save_dict(dict_representation, self._representation_type())

        return self.store_id

    @abc.abstractmethod
    def _representation_type(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def load_all_fields(self, file_pers_service: AbstractFilePersistenceService,
                        dict_pers_service: AbstractDictPersistenceService, restore_root: str,
                        load_recursive: bool = True, load_files: bool = True):
        """
        Loads all fields that have not been loaded so far.
        :param file_pers_service: An instance of AbstractFilePersistenceService that is used to store files.
        :param dict_pers_service: An instance of AbstractDictPersistenceService that is used to store metadata as dicts.
        :param restore_root: The path where restored files are stored to.
        :param load_recursive: If set to True all referenced objects are loaded fully,
        if set to False (default) only the references are restored
        :param load_files: If True all referenced files are loaded, if False only id is loaded.
        :return:
        """

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService, restore_root: str, load_recursive: bool = False,
             load_files: bool = False):
        """
        Loads the schema object from database/disk.
        :param obj_id: The identifier for the SchemaObj in the database/disk.
        :param file_pers_service: An instance of AbstractFilePersistenceService that is used to store files.
        :param dict_pers_service: An instance of AbstractDictPersistenceService that is used to store metadata as dicts.
        :param restore_root: The path where restored files are stored to.
        :param load_recursive: If set to True all referenced objects are loaded fully,
        if set to False (default) only the references are restored
        :param load_files: If True all referenced files are loaded, if False only id is loaded.
        """

        instance = cls.load_placeholder(obj_id)
        instance.load_all_fields(file_pers_service, dict_pers_service, restore_root, load_recursive, load_files)

        return instance

    @classmethod
    def load_placeholder(cls, obj_id: str):
        """
        Loads the schema object from database/disk.
        :param obj_id: The identifier for the SchemaObj in the database/disk.
        """
        return cls(store_id=obj_id)

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
        raise NotImplementedError

    @abc.abstractmethod
    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        raise NotImplementedError

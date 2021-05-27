import abc
import tempfile

from mmlib.constants import ID
from mmlib.persistence import FilePersistenceService, DictPersistenceService

METADATA_SIZE = 'metadata_size'


class SchemaObj(metaclass=abc.ABCMeta):

    def __init__(self, store_id: str = None):
        self.store_id = store_id

    @classmethod
    def load_placeholder(cls, obj_id: str):
        """
        Loads the schema object from database/disk.
        :param obj_id: The identifier for the SchemaObj in the database/disk.
        """
        return cls(store_id=obj_id)

    @classmethod
    def load(cls, obj_id: str, file_pers_service: FilePersistenceService,
             dict_pers_service: DictPersistenceService, restore_root: str, load_recursive: bool = False,
             load_files: bool = False):
        """
        Loads the schema object from database/disk.
        :param obj_id: The identifier for the SchemaObj in the database/disk.
        :param file_pers_service: An instance of FilePersistenceService that is used to store files.
        :param dict_pers_service: An instance of DictPersistenceService that is used to store metadata as dicts.
        :param restore_root: The path where restored files are stored to.
        :param load_recursive: If set to True all referenced objects are loaded fully,
        if set to False (default) only the references are restored
        :param load_files: If True all referenced files are loaded, if False only id is loaded.
        """

        instance = cls.load_placeholder(obj_id)
        instance.load_all_fields(file_pers_service, dict_pers_service, restore_root, load_recursive, load_files)

        return instance

    def persist(self, file_pers_service: FilePersistenceService,
                dict_pers_service: DictPersistenceService) -> str:
        """
        Persists the schema object.
        :param file_pers_service: An instance of FilePersistenceService that is used to store files.
        :param dict_pers_service: An instance of DictPersistenceService that is used to store metadata as dicts.
        """
        if self.store_id and dict_pers_service.id_exists(self.store_id, self._representation_type):
            # if the id already exists, we do not have to persist again
            return self.store_id

        if not self.store_id:
            self.store_id = dict_pers_service.generate_id()

        dict_representation = {
            ID: self.store_id,
        }

        self._persist_class_specific_fields(dict_representation, file_pers_service, dict_pers_service)

        dict_pers_service.save_dict(dict_representation, self._representation_type)

        return self.store_id

    @abc.abstractmethod
    def load_all_fields(self, file_pers_service: FilePersistenceService,
                        dict_pers_service: DictPersistenceService, restore_root: str,
                        load_recursive: bool = True, load_files: bool = True):
        """
        Loads all fields that have not been loaded so far.
        :param file_pers_service: An instance of FilePersistenceService that is used to store files.
        :param dict_pers_service: An instance of DictPersistenceService that is used to store metadata as dicts.
        :param restore_root: The path where restored files are stored to.
        :param load_recursive: If set to True all referenced objects are loaded fully,
        if set to False (default) only the references are restored
        :param load_files: If True all referenced files are loaded, if False only id is loaded.
        :return:
        """

    def size_info(self, file_pers_service: FilePersistenceService,
                  dict_pers_service: DictPersistenceService) -> dict:
        """
        Calculates and returns a size info dict of the schema obj. All numbers are in in bytes.
        :param file_pers_service: An instance of FilePersistenceService that is used to store and load files.
        :param dict_pers_service: An instance of DictPersistenceService that is used to store and load metadata
         as dicts.
        :return: Dict giving detailed size information in bytes bytes.
        """

        size_dict = {METADATA_SIZE: dict_pers_service.dict_size(self.store_id, self._representation_type)}

        # size of reference_fields
        with tempfile.TemporaryDirectory() as tmp_path:
            self.load_all_fields(file_pers_service, dict_pers_service, tmp_path, False, False)
            self._add_reference_sizes(size_dict, file_pers_service, dict_pers_service)

        # return {self._representation_type: size_dict}
        return size_dict

    @abc.abstractmethod
    def _add_reference_sizes(self, size_dict, file_pers_service, dict_pers_service):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _representation_type(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        raise NotImplementedError

import abc

from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from schema.schema_obj import SchemaObj
from util.init_from_file import create_object

ID = 'id'
CODE_FILE = 'code_file'
CLASS_NAME = 'class_name'
STATE_FILE = 'state_file'

RESTORABLE_OBJECT = 'restorable_object'


class RestorableObjectWrapper(SchemaObj):

    def __init__(self, instance: object, code: str, class_name: str, state_file: str, store_id: str = None):
        self.store_id = store_id
        self.instance = instance
        self.code = code
        self.class_name = class_name
        self.state_file = state_file

    def persist(self, file_pers_service: AbstractFilePersistenceService,
                dict_pers_service: AbstractDictPersistenceService) -> str:
        if not self.store_id:
            self.store_id = dict_pers_service.generate_id()

        # save the state of the internal instance
        self._save_instance_state()
        state_file_id = file_pers_service.save_file(self.state_file)
        code_file_id = file_pers_service.save_file(self.code)

        dict_representation = {
            ID: self.store_id,
            CODE_FILE: code_file_id,
            CLASS_NAME: self.class_name,
            STATE_FILE: state_file_id
        }

        dict_pers_service.save_dict(dict_representation, RESTORABLE_OBJECT)

        return self.store_id

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService, restore_root: str):
        restored_dict = dict_pers_service.recover_dict(obj_id, RESTORABLE_OBJECT)

        code_file_id = restored_dict[CODE_FILE]
        code_file_path = file_pers_service.recover_file(code_file_id, restore_root)
        class_name = restored_dict[CLASS_NAME]
        state_file_id = restored_dict[STATE_FILE]
        state_file_path = file_pers_service.recover_file(state_file_id, restore_root)

        instance = create_object(code_file_path, class_name)

        restorable_obj_wrapper = cls(instance=instance, store_id=obj_id, code=code_file_path, class_name=class_name,
                                     state_file=state_file_path)

        restorable_obj_wrapper._restore_instance_from_state()

        return restorable_obj_wrapper

    def size_in_bytes(self, file_pers_service: AbstractFilePersistenceService,
                      dict_pers_service: AbstractDictPersistenceService) -> int:
        result = 0

        # size of the dict
        result += dict_pers_service.dict_size(self.store_id, RESTORABLE_OBJECT)
        restored_dict = dict_pers_service.recover_dict(self.store_id, RESTORABLE_OBJECT)

        # size of all referenced files/objects
        result += file_pers_service.file_size(restored_dict[CODE_FILE])
        result += file_pers_service.file_size(restored_dict[STATE_FILE])

        return result

    @abc.abstractmethod
    def _save_instance_state(self):
        """
        Saves the state of the internal instance to a file.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _restore_instance_from_state(self):
        """
        Loads the state for the internal instance from a file.
        """
        raise NotImplementedError

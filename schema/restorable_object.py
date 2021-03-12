import abc

from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from schema.schema_obj import SchemaObj
from util.init_from_file import create_object_with_parameters

ID = 'id'
CODE_FILE = 'code_file'
CLASS_NAME = 'class_name'
STATE_FILE = 'state_file'
INIT_ARGS = 'init_args'
INIT_REF_TYPE_ARGS = 'init_ref_type_args'

RESTORABLE_OBJECT = 'restorable_object'


class RestorableObjectWrapper(SchemaObj):

    def __init__(self, code: str, class_name: str, init_args: dict, init_ref_type_args: [str], instance: object = None,
                 state_file: str = None, store_id: str = None):
        self.store_id = store_id
        self.instance = instance
        self.code = code
        self.class_name = class_name
        self.init_args = init_args
        self.init_ref_type_args = init_ref_type_args
        self.state_file = state_file

    def set_instance(self, instance):
        self.instance = instance

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
            INIT_ARGS: self.init_args,
            INIT_REF_TYPE_ARGS: self.init_ref_type_args,
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
        init_args = restored_dict[INIT_ARGS]
        ref_type_args = restored_dict[INIT_REF_TYPE_ARGS]
        state_file_id = restored_dict[STATE_FILE]
        state_file_path = file_pers_service.recover_file(state_file_id, restore_root)

        restorable_obj_wrapper = cls(store_id=obj_id, code=code_file_path, class_name=class_name,
                                     init_args=init_args, init_ref_type_args=ref_type_args, state_file=state_file_path)

        restorable_obj_wrapper._restore_instance_state()

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

    def restore_instance(self, ref_type_args: dict):
        keys = set(ref_type_args.keys())
        assert keys == set(self.init_ref_type_args), 'not all parameters are given'

        self.instance = create_object_with_parameters(code=self.code, class_name=self.class_name,
                                                      init_args=self.init_args, init_ref_type_args=ref_type_args)

        self._restore_instance_state()

    @abc.abstractmethod
    def _save_instance_state(self):
        """
        Saves the state of the internal instance to a file. Only needs to be implemented when there is a internal state
        that can not be reproduced by passing the right arguments in the constructor.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _restore_instance_state(self):
        """
        Loads the state for the internal instance from a file.
        """
        raise NotImplementedError

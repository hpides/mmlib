import abc
import configparser

from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from schema.schema_obj import SchemaObj
from util.init_from_file import create_object_with_parameters

IMPORT_CMD = 'import_cmd'

ID = 'id'
CODE_FILE = 'code_file'
CLASS_NAME = 'class_name'
STATE_FILE = 'state_file'
INIT_ARGS = 'init_args'
CONFIG_ARGS = 'config_args'
INIT_REF_TYPE_ARGS = 'init_ref_type_args'

RESTORABLE_OBJECT = 'restorable_object'


def add_params_from_config(init_args, config_args):
    # TODO think how to nicely integrate the config file
    config_file = '/Users/nils/Studium/master-thesis/mmlib/tests/config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)

    for k, v in config_args.items():
        init_args[k] = config['VALUES'][v]


class RestorableObjectWrapper(SchemaObj):

    def __init__(self, class_name: str, init_args: dict, init_ref_type_args: [str], config_args: dict, code: str = None,
                 import_cmd: str = None, instance: object = None, store_id: str = None):

        self.store_id = store_id
        self.instance = instance
        self.code = code
        self.import_cmd = import_cmd
        self.class_name = class_name
        self.init_args = init_args
        self.config_args = config_args
        self.init_ref_type_args = init_ref_type_args

    def set_instance(self, instance):
        self.instance = instance

    def persist(self, file_pers_service: AbstractFilePersistenceService,
                dict_pers_service: AbstractDictPersistenceService) -> str:
        if not self.store_id:
            self.store_id = dict_pers_service.generate_id()

        dict_representation = {
            ID: self.store_id,
            CLASS_NAME: self.class_name,
            INIT_ARGS: self.init_args,
            CONFIG_ARGS: self.config_args,
            INIT_REF_TYPE_ARGS: self.init_ref_type_args,
        }

        self._add_optional_fields(dict_representation, file_pers_service, dict_pers_service)

        dict_pers_service.save_dict(dict_representation, RESTORABLE_OBJECT)

        return self.store_id

    def _add_optional_fields(self, dict_representation, file_pers_service, dict_pers_service):
        if self.code:
            assert self.import_cmd is None, 'if code is set then there should be no import cmd'
            code_file_id = file_pers_service.save_file(self.code)
            dict_representation[CODE_FILE] = code_file_id
        if self.import_cmd:
            assert self.code is None, 'if import_cmd is set then there should be no code'
            dict_representation[IMPORT_CMD] = self.import_cmd

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService, restore_root: str):
        restored_dict = dict_pers_service.recover_dict(obj_id, RESTORABLE_OBJECT)

        class_name = restored_dict[CLASS_NAME]
        init_args = restored_dict[INIT_ARGS]
        config_args = restored_dict[CONFIG_ARGS]
        ref_type_args = restored_dict[INIT_REF_TYPE_ARGS]

        code_file_path = None
        if CODE_FILE in restored_dict:
            code_file_id = restored_dict[CODE_FILE]
            code_file_path = file_pers_service.recover_file(code_file_id, restore_root)

        import_cmd = restored_dict[IMPORT_CMD] if IMPORT_CMD in restored_dict else None

        restorable_obj_wrapper = cls(store_id=obj_id, code=code_file_path, class_name=class_name, import_cmd=import_cmd,
                                     init_args=init_args, init_ref_type_args=ref_type_args, config_args=config_args)

        return restorable_obj_wrapper

    def size_in_bytes(self, file_pers_service: AbstractFilePersistenceService,
                      dict_pers_service: AbstractDictPersistenceService) -> int:
        result = 0

        # size of the dict
        result += dict_pers_service.dict_size(self.store_id, RESTORABLE_OBJECT)
        restored_dict = dict_pers_service.recover_dict(self.store_id, RESTORABLE_OBJECT)

        # size of all referenced files/objects
        # TODO adjust method, code not always given
        result += file_pers_service.file_size(restored_dict[CODE_FILE])
        result += file_pers_service.file_size(restored_dict[STATE_FILE])

        return result

    def restore_instance(self, ref_type_args: dict = None):
        if self.init_ref_type_args or ref_type_args:
            assert self.init_ref_type_args and ref_type_args, self._generate_non_matching_parameter_message(
                ref_type_args)
            keys = set(ref_type_args.keys())
            assert keys == set(self.init_ref_type_args), self._generate_non_matching_parameter_message(
                ref_type_args)

        init_args = self.init_args
        add_params_from_config(init_args, self.config_args)

        self.instance = create_object_with_parameters(
            code=self.code, import_cmd=self.import_cmd, class_name=self.class_name, init_args=init_args,
            init_ref_type_args=ref_type_args)

    def _generate_non_matching_parameter_message(self, ref_type_args):
        return 'given parameters not match the expected parameters - expected: {}, given: {}'.format(
            self.init_ref_type_args, ref_type_args)


class StateFileRestorableObjectWrapper(RestorableObjectWrapper):

    def __init__(self, class_name: str, init_args: dict, init_ref_type_args: [str], code: str = None,
                 import_cmd: str = None, instance: object = None, store_id: str = None, state_file: str = None):
        super().__init__(class_name, init_args, init_ref_type_args, code, import_cmd, instance, store_id)
        self.state_file = state_file

    def persist(self, file_pers_service: AbstractFilePersistenceService,
                dict_pers_service: AbstractDictPersistenceService) -> str:
        return_id = super(StateFileRestorableObjectWrapper, self).persist(file_pers_service, dict_pers_service)

        self._save_instance_state()

        return return_id

    def _add_optional_fields(self, dict_representation, file_pers_service, dict_pers_service):
        super(StateFileRestorableObjectWrapper, self)._add_optional_fields(dict_representation, file_pers_service,
                                                                           dict_pers_service)

        if self.state_file:
            state_file_id = file_pers_service.save_file(self.state_file)
            dict_representation[STATE_FILE] = state_file_id

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService, restore_root: str):
        obj = StateFileRestorableObjectWrapper.load(obj_id, file_pers_service, dict_pers_service, restore_root)

        obj._restore_instance_state()

        return obj

    def restore_instance(self, ref_type_args: dict = None):
        super(StateFileRestorableObjectWrapper, self).restore_instance(ref_type_args)

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

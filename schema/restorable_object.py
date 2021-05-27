import abc
import configparser
import os
import tempfile
from abc import ABCMeta
from typing import Dict

import torch

from mmlib.constants import MMLIB_CONFIG, VALUES, ID
from mmlib.persistence import FilePersistenceService, DictPersistenceService
from schema.file_reference import FileReference
from schema.schema_obj import SchemaObj
from util.helper import class_name, source_file
from util.init_from_file import create_object_with_parameters

STATE_DICT = 'state_dict'

CODE_FILE = 'code_file'
CLASS_NAME = 'class_name'
IMPORT_CMD = 'import_cmd'
INIT_ARGS = 'init_args'
CONFIG_ARGS = 'config_args'
INIT_REF_TYPE_ARGS = 'init_ref_type_args'
STATE_FILE = 'state_file'

RESTORABLE_OBJECT = 'restorable_object'


class RestoredModelInfo:
    def __init__(self, model: torch.nn.Module):
        self.model = model


class AbstractRestorableObjectWrapper(SchemaObj, metaclass=ABCMeta):

    def __init__(self, c_name: str, code: FileReference, import_cmd: str = None, instance: object = None,
                 store_id: str = None):
        super().__init__(store_id)
        self.instance = instance
        assert isinstance(code, FileReference) or code is None
        self.import_cmd = import_cmd
        self.class_name = c_name
        if self.class_name is None and instance:
            self.class_name = class_name(instance)
        self.code = code
        if self.code is None and self.import_cmd is None and instance:
            self.code = code if code else FileReference(path=source_file(instance))

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):

        # mandatory fields
        dict_representation[CLASS_NAME] = self.class_name

        # optional fields
        if self.code:
            file_pers_service.save_file(self.code)
            dict_representation[CODE_FILE] = self.code.reference_id
        if self.import_cmd:
            dict_representation[IMPORT_CMD] = self.import_cmd

    def _add_reference_sizes(self, size_dict, file_pers_service, dict_pers_service):
        if self.code:
            file_pers_service.file_size(self.code)
            size_dict[CODE_FILE] = self.code.size

    @property
    def _representation_type(self) -> str:
        return RESTORABLE_OBJECT

    @classmethod
    def load_specific_placeholder(cls, obj_id: str, dict_pers_service: DictPersistenceService):
        """
        Loads the schema object from database/disk.
        :param obj_id: The identifier for the SchemaObj in the database/disk.
        :param dict_pers_service: An instance of DictPersistenceService that is used to store metadata as dicts.

        """
        restored_dict = dict_pers_service.recover_dict(obj_id, RESTORABLE_OBJECT)
        if STATE_FILE in restored_dict:
            return StateFileRestorableObjectWrapper(store_id=obj_id)
        else:
            return RestorableObjectWrapper(store_id=obj_id)


class RestorableObjectWrapper(AbstractRestorableObjectWrapper):

    def __init__(self, c_name: str = None, init_args: dict = None, init_ref_type_args: [str] = None,
                 config_args: dict = None, code: FileReference = None, import_cmd: str = None, instance: object = None,
                 store_id: str = None):
        super().__init__(c_name=c_name, code=code, import_cmd=import_cmd, instance=instance, store_id=store_id)

        self.init_args = init_args if init_args else {}
        self.config_args = config_args if config_args else {}
        self.init_ref_type_args = init_ref_type_args if init_ref_type_args else []

    def set_instance(self, instance):
        self.instance = instance

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        super()._persist_class_specific_fields(dict_representation, file_pers_service, dict_pers_service)

        dict_representation[INIT_ARGS] = self.init_args
        dict_representation[CONFIG_ARGS] = self.config_args
        dict_representation[INIT_REF_TYPE_ARGS] = self.init_ref_type_args

    def load_all_fields(self, file_pers_service: FilePersistenceService,
                        dict_pers_service: DictPersistenceService, restore_root: str,
                        load_recursive: bool = True, load_files: bool = True):
        restored_dict = dict_pers_service.recover_dict(self.store_id, RESTORABLE_OBJECT)

        self.class_name, self.config_args, self.import_cmd, self.init_args, self.init_ref_type_args = \
            _restore_non_ref_fields(restored_dict)

        self.code = _restore_code(file_pers_service, restore_root, restored_dict, load_files)

    def restore_instance(self, ref_type_args: dict = None):
        if self.init_ref_type_args or ref_type_args:
            assert self.init_ref_type_args and ref_type_args, self._generate_non_matching_parameter_message(
                ref_type_args)
            keys = set(ref_type_args.keys())
            assert keys == set(self.init_ref_type_args), self._generate_non_matching_parameter_message(
                ref_type_args)

        init_args = self.init_args
        if len(self.config_args) > 0:
            add_params_from_config(init_args, self.config_args)

        code_path = self.code.path if self.code else None
        self.instance = create_object_with_parameters(
            code_file=code_path, import_cmd=self.import_cmd, class_name=self.class_name, init_args=init_args,
            init_ref_type_args=ref_type_args)

    def _generate_non_matching_parameter_message(self, ref_type_args):
        return 'given parameters not match the expected parameters - expected: {}, given: {}'.format(
            self.init_ref_type_args, ref_type_args)


def _restore_non_ref_fields(restored_dict):
    class_name = restored_dict[CLASS_NAME]
    init_args = restored_dict[INIT_ARGS]
    config_args = restored_dict[CONFIG_ARGS]
    ref_type_args = restored_dict[INIT_REF_TYPE_ARGS]
    import_cmd = restored_dict[IMPORT_CMD] if IMPORT_CMD in restored_dict else None
    return class_name, config_args, import_cmd, init_args, ref_type_args


def _restore_code(file_pers_service, restore_root, restored_dict, load_files):
    code_file = None

    if CODE_FILE in restored_dict:
        code_file_id = restored_dict[CODE_FILE]
        code_file = FileReference(reference_id=code_file_id)

        if load_files:
            file_pers_service.recover_file(code_file, restore_root)

    return code_file


class StateDictObj(metaclass=abc.ABCMeta):
    def __init__(self):
        self.state_objs: Dict[str, RestorableObjectWrapper] = {}


class StateDictRestorableObjectWrapper(AbstractRestorableObjectWrapper):

    def __init__(self, c_name: str = None, code: FileReference = None, instance: StateDictObj = None, state_objs=None,
                 store_id: str = None):
        super().__init__(c_name=c_name, code=code, instance=instance, store_id=store_id)
        self.instance: StateDictObj = instance
        self.state_objs = state_objs

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        super()._persist_class_specific_fields(dict_representation, file_pers_service, dict_pers_service)

        # persist instance state dict
        state_dict_refs = {}
        for k, v in self.instance.state_objs.items():
            obj: RestorableObjectWrapper = v
            obj_id = obj.persist(file_pers_service, dict_pers_service)
            state_dict_refs[k] = obj_id

        dict_representation[STATE_DICT] = state_dict_refs

    @classmethod
    def load(cls, obj_id: str, file_pers_service: FilePersistenceService,
             dict_pers_service: DictPersistenceService, restore_root: str, load_recursive: bool = False,
             load_files: bool = False):
        restored_dict = dict_pers_service.recover_dict(obj_id, RESTORABLE_OBJECT)

        c_name = restored_dict[CLASS_NAME]
        code_file_path = _restore_code(file_pers_service, restore_root, restored_dict, load_files)
        state_objs = restored_dict[STATE_DICT]

        restorable_obj_wrapper = cls(store_id=obj_id, code=code_file_path, c_name=c_name, state_objs=state_objs)

        return restorable_obj_wrapper

    def load_all_fields(self, file_pers_service: FilePersistenceService,
                        dict_pers_service: DictPersistenceService, restore_root: str,
                        load_recursive: bool = True, load_files: bool = True):
        restored_dict = dict_pers_service.recover_dict(self.store_id, RESTORABLE_OBJECT)

        self.class_name = restored_dict[CLASS_NAME]
        self.code = _restore_code(file_pers_service, restore_root, restored_dict, load_files)
        self.state_objs = restored_dict[STATE_DICT]

    @abc.abstractmethod
    def restore_instance(self, file_pers_service: FilePersistenceService,
                         dict_pers_service: DictPersistenceService, restore_root: str):
        raise NotImplementedError

    def _add_reference_sizes(self, size_dict, file_pers_service, dict_pers_service):
        s_dict = {}
        for k, v in self.state_objs.items():
            place_holder = AbstractRestorableObjectWrapper.load_specific_placeholder(v, dict_pers_service)
            s_dict[k] = place_holder.size_info(file_pers_service, dict_pers_service)

        size_dict[STATE_DICT] = s_dict


class StateFileRestorableObject(StateDictObj):

    @abc.abstractmethod
    def save_instance_state(self, path):
        """
        Saves the instance state to a file. The file is saved at the given path.
        :param path: The path to save the state file to.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def restore_instance_state(self, path):
        """
        Restores the instance state from the given file.
        :param path: The path to the file that holds the information to recover the insatcne state.
        """
        raise NotImplementedError


class StateFileRestorableObjectWrapper(RestorableObjectWrapper):

    def __init__(self, c_name: str = None, init_args: dict = None, init_ref_type_args: [str] = None,
                 config_args: dict = None, code: FileReference = None, import_cmd: str = None,
                 instance: StateFileRestorableObject = None, store_id: str = None, state_file: FileReference = None):
        super().__init__(c_name=c_name, init_args=init_args, init_ref_type_args=init_ref_type_args,
                         config_args=config_args, code=code, import_cmd=import_cmd, instance=instance,
                         store_id=store_id)
        self.state_file = state_file

    # # TODO check if we need this method
    def persist(self, file_pers_service: FilePersistenceService,
                dict_pers_service: DictPersistenceService) -> str:

        # the state of the instance has probably changed -> need to store new version with new id
        self.store_id = dict_pers_service.generate_id()

        dict_representation = {
            ID: self.store_id,
        }

        self._persist_class_specific_fields(dict_representation, file_pers_service, dict_pers_service)

        dict_pers_service.save_dict(dict_representation, self._representation_type)

        return self.store_id

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):

        super()._persist_class_specific_fields(dict_representation, file_pers_service, dict_pers_service)

        if self.instance:
            with tempfile.TemporaryDirectory() as tmp_path:
                state_file = FileReference(path=os.path.join(tmp_path, 'state'))
                self.instance.save_instance_state(state_file.path)
                file_pers_service.save_file(state_file)

            dict_representation[STATE_FILE] = state_file.reference_id

    @classmethod
    def load(cls, obj_id: str, file_pers_service: FilePersistenceService,
             dict_pers_service: DictPersistenceService, restore_root: str, load_recursive: bool = False,
             load_files: bool = False):
        restored_dict = dict_pers_service.recover_dict(obj_id, RESTORABLE_OBJECT)

        class_name, config_args, import_cmd, init_args, ref_type_args = _restore_non_ref_fields(restored_dict)
        code_file = _restore_code(file_pers_service, restore_root, restored_dict, load_files)

        state_file = _recover_state_file(file_pers_service, load_files, restore_root, restored_dict)

        obj = cls(store_id=obj_id, config_args=config_args, import_cmd=import_cmd, init_args=init_args,
                  c_name=class_name, code=code_file, init_ref_type_args=ref_type_args, state_file=state_file)

        return obj

    def load_all_fields(self, file_pers_service: FilePersistenceService,
                        dict_pers_service: DictPersistenceService, restore_root: str,
                        load_recursive: bool = True, load_files: bool = True):

        restored_dict = dict_pers_service.recover_dict(self.store_id, RESTORABLE_OBJECT)

        super().load_all_fields(file_pers_service, dict_pers_service, restore_root, load_recursive, load_files)
        self.state_file = _recover_state_file(file_pers_service, load_files, restore_root, restored_dict)

    def restore_instance(self, ref_type_args: dict = None):
        super(StateFileRestorableObjectWrapper, self).restore_instance(ref_type_args)

        if self.state_file:
            self.instance.restore_instance_state(self.state_file.path)

    def _add_reference_sizes(self, size_dict, file_pers_service, dict_pers_service):
        super()._add_reference_sizes(size_dict, file_pers_service, dict_pers_service)
        file_pers_service.file_size(self.state_file)
        size_dict[STATE_FILE] = self.state_file.size


def _recover_state_file(file_pers_service, load_files, restore_root, restored_dict):
    state_file = None

    if STATE_FILE in restored_dict:
        state_file_id = restored_dict[STATE_FILE]
        state_file = FileReference(reference_id=state_file_id)

        if load_files:
            file_pers_service.recover_file(state_file, restore_root)

    return state_file


class TrainService(StateDictObj):

    @abc.abstractmethod
    def train(self, model: torch.nn.Module):
        raise NotImplementedError


def add_params_from_config(init_args, config_args):
    config_file = os.getenv(MMLIB_CONFIG)
    config = configparser.ConfigParser()
    config.read(config_file)

    for k, v in config_args.items():
        init_args[k] = config[VALUES][v]

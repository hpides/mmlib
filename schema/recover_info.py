import abc
import configparser
import os

from mmlib.constants import MMLIB_CONFIG, CURRENT_DATA_ROOT, VALUES
from mmlib.persistence import FilePersistenceService, DictPersistenceService
from schema.dataset import Dataset
from schema.environment import Environment
from schema.file_reference import FileReference
from schema.schema_obj import SchemaObj
from schema.train_info import TrainInfo
from util.helper import copy_all_data, clean

RECOVER_INFO = 'recover_info'


class AbstractRecoverInfo(SchemaObj, metaclass=abc.ABCMeta):

    @property
    def _representation_type(self) -> str:
        return RECOVER_INFO


MODEL_CODE = 'model_code'
MODEL_CLASS_NAME = 'model_class_name'
PARAMETERS = 'parameters'
ENVIRONMENT = 'environment'


class FullModelRecoverInfo(AbstractRecoverInfo):

    def __init__(self, parameters_file: FileReference = None, model_code: FileReference = None,
                 model_class_name: str = None, environment: Environment = None, store_id: str = None):
        super().__init__(store_id)
        self.model_code = model_code
        self.model_class_name = model_class_name
        self.parameters_file = parameters_file
        self.environment = environment

    def load_all_fields(self, file_pers_service: FilePersistenceService,
                        dict_pers_service: DictPersistenceService, restore_root: str,
                        load_recursive: bool = True, load_files: bool = True):
        restored_dict = dict_pers_service.recover_dict(self.store_id, RECOVER_INFO)

        self.model_class_name = restored_dict[MODEL_CLASS_NAME]

        self.model_code = _recover_model_code(file_pers_service, load_files, restore_root, restored_dict)
        self.parameters_file = _recover_parameters(file_pers_service, load_files, restore_root, restored_dict)
        self.environment = _recover_environment(dict_pers_service, file_pers_service, load_recursive, restore_root,
                                                restored_dict)

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        file_pers_service.save_file(self.model_code)
        file_pers_service.save_file(self.parameters_file)
        env_id = self.environment.persist(file_pers_service, dict_pers_service)

        dict_representation[PARAMETERS] = self.parameters_file.reference_id
        dict_representation[MODEL_CODE] = self.model_code.reference_id
        dict_representation[MODEL_CLASS_NAME] = self.model_class_name
        dict_representation[ENVIRONMENT] = env_id

    def _add_reference_sizes(self, size_dict, file_pers_service, dict_pers_service):
        size_dict[ENVIRONMENT] = self.environment.size_info(file_pers_service, dict_pers_service)

        file_pers_service.file_size(self.model_code)
        size_dict[MODEL_CODE] = self.model_code.size

        file_pers_service.file_size(self.parameters_file)
        size_dict[PARAMETERS] = self.parameters_file.size


def _recover_parameters(file_pers_service, load_files, restore_root, restored_dict):
    parameterss_file_id = restored_dict[PARAMETERS]
    parameters_file = FileReference(reference_id=parameterss_file_id)

    if load_files:
        file_pers_service.recover_file(parameters_file, restore_root)

    return parameters_file


def _recover_environment(dict_pers_service, file_pers_service, load_recursive, restore_root, restored_dict):
    env_id = restored_dict[ENVIRONMENT]
    if load_recursive:
        env = Environment.load(env_id, file_pers_service, dict_pers_service, restore_root)
    else:
        env = Environment.load_placeholder(env_id)
    return env


UPDATE = 'update'
UPDATE_TYPE = 'update_type'
INDEPENDENT = 'independent'


class WeightsUpdateRecoverInfo(AbstractRecoverInfo):

    def __init__(self, update: FileReference = None, update_type: str = None, independent: bool = None,
                 store_id: str = None):
        super().__init__(store_id)
        self.update = update
        self.update_type = update_type
        self.independent = independent

    def load_all_fields(self, file_pers_service: FilePersistenceService, dict_pers_service: DictPersistenceService,
                        restore_root: str, load_recursive: bool = True, load_files: bool = True):
        restored_dict = dict_pers_service.recover_dict(self.store_id, RECOVER_INFO)

        self.update = _restore_update(file_pers_service, load_files, restore_root, restored_dict)
        self.update_type = restored_dict[UPDATE_TYPE]
        self.independent = restored_dict[INDEPENDENT]

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        file_pers_service.save_file(self.update)
        dict_representation[UPDATE] = self.update.reference_id
        dict_representation[UPDATE_TYPE] = self.update_type
        dict_representation[INDEPENDENT] = self.independent

    def _add_reference_sizes(self, size_dict, file_pers_service, dict_pers_service):
        file_pers_service.file_size(self.update)
        size_dict[UPDATE] = self.update.size


def _restore_update(file_pers_service, load_files, restore_root, restored_dict):
    update_id = restored_dict[UPDATE]
    update = FileReference(reference_id=update_id)

    if load_files:
        file_pers_service.recover_file(update, restore_root)

    return update


DATASET = 'dataset'
TRAIN_INFO = 'train_info'


class ProvenanceRecoverInfo(AbstractRecoverInfo):

    def __init__(self, dataset: Dataset = None, train_info: TrainInfo = None, environment: Environment = None,
                 store_id: str = None):
        super().__init__(store_id)
        self.dataset = dataset
        self.train_info = train_info
        self.environment = environment

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        dataset_id = self.dataset.persist(file_pers_service, dict_pers_service)
        env_id = self.environment.persist(file_pers_service, dict_pers_service)
        train_info_id = self.train_info.persist(file_pers_service, dict_pers_service)

        dict_representation[DATASET] = dataset_id
        dict_representation[TRAIN_INFO] = train_info_id
        dict_representation[ENVIRONMENT] = env_id

    def load_all_fields(self, file_pers_service: FilePersistenceService,
                        dict_pers_service: DictPersistenceService, restore_root: str,
                        load_recursive: bool = True, load_files: bool = True):
        restored_dict = dict_pers_service.recover_dict(self.store_id, RECOVER_INFO)

        dataset_id = restored_dict[DATASET]
        self.dataset = _recover_data(dataset_id, dict_pers_service, file_pers_service, load_files, load_recursive,
                                     restore_root)

        self.train_info = _restore_train_info(
            dict_pers_service, file_pers_service, restore_root, restored_dict, load_recursive, load_files)

        self.environment = _recover_environment(dict_pers_service, file_pers_service, load_recursive, restore_root,
                                                restored_dict)

    def _add_reference_sizes(self, size_dict, file_pers_service, dict_pers_service):
        size_dict[ENVIRONMENT] = self.environment.size_info(file_pers_service, dict_pers_service)
        size_dict[TRAIN_INFO] = self.train_info.size_info(file_pers_service, dict_pers_service)
        size_dict[DATASET] = self.dataset.size_info(file_pers_service, dict_pers_service)


def _data_dst_path():
    config_file = os.getenv(MMLIB_CONFIG)
    config = configparser.ConfigParser()
    config.read(config_file)

    return config[VALUES][CURRENT_DATA_ROOT]


def _recover_data(dataset_id, dict_pers_service, file_pers_service, load_files, load_recursive, restore_root):
    dataset = Dataset.load(dataset_id, file_pers_service, dict_pers_service, restore_root, load_recursive,
                           load_files)
    # make data available for train_info
    if load_files:
        # TODO for now we copy the data, maybe if we run into performance issues we should use move instead of copy
        data_dst_path = _data_dst_path()
        clean(data_dst_path)
        copy_all_data(dataset.raw_data.path, data_dst_path)
    return dataset


def _recover_model_code(file_pers_service, load_files, restore_root, restored_dict):
    model_code_id = restored_dict[MODEL_CODE]
    model_code = FileReference(reference_id=model_code_id)

    if load_files:
        file_pers_service.recover_file(model_code, restore_root)

    return model_code


def _restore_train_info(dict_pers_service, file_pers_service, restore_root, restored_dict, load_recursive,
                        load_files):
    train_info_id = restored_dict[TRAIN_INFO]
    if not load_recursive:
        train_info = TrainInfo.load_placeholder(train_info_id)
    else:
        train_info = TrainInfo.load(
            train_info_id, file_pers_service, dict_pers_service, restore_root, load_recursive, load_files)
    return train_info

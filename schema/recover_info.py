import abc
import configparser
import os

from mmlib.constants import MMLIB_CONFIG, CURRENT_DATA_ROOT, VALUES, ID
from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from schema.dataset import Dataset
from schema.schema_obj import SchemaObj
from schema.train_info import TrainInfo
from util.helper import copy_all_data, clean

TRAIN_INFO = 'train_info'

DATASET = 'dataset'


class AbstractRecoverInfo(SchemaObj, metaclass=abc.ABCMeta):

    def __init__(self, model_code: str, model_class_name: str, store_id: str = None):
        super().__init__(store_id)
        self.model_code_file_path = model_code
        self.model_class_name = model_class_name

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        model_code_id = file_pers_service.save_file(self.model_code_file_path)

        dict_representation[MODEL_CODE] = model_code_id
        dict_representation[MODEL_CLASS_NAME] = self.model_class_name

    def size_in_bytes(self, file_pers_service: AbstractFilePersistenceService,
                      dict_pers_service: AbstractDictPersistenceService) -> int:
        result = 0

        # size of the dict
        result += dict_pers_service.dict_size(self.store_id, RECOVER_INFO)

        # size of all referenced files/objects
        restored_dict = dict_pers_service.recover_dict(self.store_id, RECOVER_INFO)
        result += file_pers_service.file_size(restored_dict[MODEL_CODE])

        # size of subclass fields
        result += self._size_class_specific_fields(restored_dict, file_pers_service, dict_pers_service)

        return result

    def _representation_type(self) -> str:
        return RECOVER_INFO

    @abc.abstractmethod
    def _size_class_specific_fields(self, restored_dict, file_pers_service, dict_pers_service):
        raise NotImplementedError


WEIGHTS = 'weights'
MODEL_CODE = 'model_code'
MODEL_CLASS_NAME = 'model_class_name'

RECOVER_INFO = 'recover_info'


class FullModelRecoverInfo(AbstractRecoverInfo):

    def __init__(self, weights_file_path: str = None, model_code_file_path=None, model_class_name: str = None,
                 store_id: str = None):
        super().__init__(model_code_file_path, model_class_name, store_id)
        self.weights_file_path = weights_file_path

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService, restore_root: str, load_recursive: bool = False,
             load_files: bool = False):
        restored_dict = dict_pers_service.recover_dict(obj_id, RECOVER_INFO)

        store_id = restored_dict[ID]
        model_class_name = restored_dict[MODEL_CLASS_NAME]

        weights_file_path = None
        model_code_file_path = None
        if load_files:
            weights_file_id = restored_dict[WEIGHTS]
            weights_file_path = file_pers_service.recover_file(weights_file_id, restore_root)
            model_code_file_id = restored_dict[MODEL_CODE]
            model_code_file_path = file_pers_service.recover_file(model_code_file_id, restore_root)

        return cls(weights_file_path=weights_file_path, model_code_file_path=model_code_file_path,
                   model_class_name=model_class_name, store_id=store_id)

    def load_all_fields(self, file_pers_service: AbstractFilePersistenceService,
                        dict_pers_service: AbstractDictPersistenceService, restore_root: str,
                        load_recursive: bool = True, load_files: bool = True):
        restored_dict = dict_pers_service.recover_dict(self.store_id, RECOVER_INFO)

        self.model_class_name = restored_dict[MODEL_CLASS_NAME]

        if load_files:
            weights_file_id = restored_dict[WEIGHTS]
            self.weights_file_path = file_pers_service.recover_file(weights_file_id, restore_root)

            model_code_file_id = restored_dict[MODEL_CODE]
            self.model_code_file_path = file_pers_service.recover_file(model_code_file_id, restore_root)

    def _size_class_specific_fields(self, restored_dict, file_pers_service, dict_pers_service):
        result = 0

        result += file_pers_service.file_size(restored_dict[WEIGHTS])

        return result

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        super()._persist_class_specific_fields(dict_representation, file_pers_service, dict_pers_service)
        weights_id = file_pers_service.save_file(self.weights_file_path)

        dict_representation[WEIGHTS] = weights_id

    def _representation_type(self) -> str:
        return RECOVER_INFO


class ProvenanceRecoverInfo(AbstractRecoverInfo):

    def load_all_fields(self, file_pers_service: AbstractFilePersistenceService,
                        dict_pers_service: AbstractDictPersistenceService, restore_root: str,
                        load_recursive: bool = True, load_files: bool = True):
        pass

    def __init__(self, dataset: Dataset = None, model_code_file_path=None, model_class_name: str = None,
                 train_info: TrainInfo = None, store_id: str = None):
        super().__init__(model_code_file_path, model_class_name, store_id)
        self.dataset = dataset
        self.train_info = train_info

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService, restore_root: str, load_recursive: bool = False,
             load_files: bool = False):
        restored_dict = dict_pers_service.recover_dict(obj_id, RECOVER_INFO)

        store_id = restored_dict[ID]
        dataset_id = restored_dict[DATASET]
        dataset = Dataset.load(dataset_id, file_pers_service, dict_pers_service, restore_root)

        # make data available for train_info
        # TODO for now we copy the data, maybe if we run into performance issues we should use move instead of copy

        data_dst_path = _data_dst_path()
        clean(data_dst_path)
        copy_all_data(dataset.raw_data, data_dst_path)

        model_code_id = restored_dict[MODEL_CODE]
        model_code_file_path = file_pers_service.recover_file(model_code_id, restore_root)
        model_class_name = restored_dict[MODEL_CLASS_NAME]
        train_info_id = restored_dict[TRAIN_INFO]
        train_info = TrainInfo.load(train_info_id, file_pers_service, dict_pers_service, restore_root)

        return cls(dataset=dataset, model_class_name=model_class_name, train_info=train_info, store_id=store_id,
                   model_code_file_path=model_code_file_path)

    def _size_class_specific_fields(self, restored_dict, file_pers_service, dict_pers_service):
        result = 0

        result += self.dataset.size_in_bytes(file_pers_service, dict_pers_service)
        result += self.train_info.size_in_bytes(file_pers_service, dict_pers_service)

        return result

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        super()._persist_class_specific_fields(dict_representation, file_pers_service, dict_pers_service)
        dataset_id = self.dataset.persist(file_pers_service, dict_pers_service)
        train_info_id = self.train_info.persist(file_pers_service, dict_pers_service)

        print('train_info_id')
        print(train_info_id)

        dict_representation[DATASET] = dataset_id
        dict_representation[TRAIN_INFO] = train_info_id


def _data_dst_path():
    # TODO magic strings
    config_file = os.getenv(MMLIB_CONFIG)
    config = configparser.ConfigParser()
    config.read(config_file)

    return config[VALUES][CURRENT_DATA_ROOT]

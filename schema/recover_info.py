import abc
import configparser
import os

from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from schema.dataset import Dataset
from schema.recover_val import RecoverVal
from schema.schema_obj import SchemaObj
from schema.train_info import TrainInfo
from util.helper import copy_all_data, clean

TRAIN_INFO = 'train_info'

DATASET = 'dataset'


class AbstractRecoverInfo(SchemaObj, metaclass=abc.ABCMeta):

    def __init__(self, store_id: str = None):
        super().__init__(store_id)

    def _representation_type(self) -> str:
        return RECOVER_INFO

    # TODO see what of teh shared functionality can be put in this abstract class
    pass


ID = 'id'
WEIGHTS = 'weights'
MODEL_CODE = 'model_code'
MODEL_CLASS_NAME = 'model_class_name'
RECOVER_VAL = 'recover_val'

RECOVER_INFO = 'recover_info'


class FullModelRecoverInfo(AbstractRecoverInfo):

    def __init__(self, weights_file_path: str, model_code_file_path, model_class_name: str, store_id: str = None,
                 recover_validation: RecoverVal = None):
        super().__init__(store_id)
        self.weights_file_path = weights_file_path
        self.model_code_file_path = model_code_file_path
        self.model_class_name = model_class_name
        self.recover_validation = recover_validation

    def persist(self, file_pers_service: AbstractFilePersistenceService,
                dict_pers_service: AbstractDictPersistenceService) -> str:

        if not self.store_id:
            self.store_id = dict_pers_service.generate_id()

        weights_id = file_pers_service.save_file(self.weights_file_path)
        model_code_id = file_pers_service.save_file(self.model_code_file_path)

        dict_representation = {
            ID: self.store_id,
            WEIGHTS: weights_id,
            MODEL_CODE: model_code_id,
            MODEL_CLASS_NAME: self.model_class_name
        }

        if self.recover_validation:
            recover_val_id = self.recover_validation.persist(file_pers_service, dict_pers_service)
            dict_representation[RECOVER_VAL] = recover_val_id

        dict_pers_service.save_dict(dict_representation, RECOVER_INFO)

        return self.store_id

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService, restore_root: str):

        restored_dict = dict_pers_service.recover_dict(obj_id, RECOVER_INFO)

        store_id = restored_dict[ID]
        weights_file_id = restored_dict[WEIGHTS]
        weights_file_path = file_pers_service.recover_file(weights_file_id, restore_root)
        model_code_file_id = restored_dict[MODEL_CODE]
        model_code_file_path = file_pers_service.recover_file(model_code_file_id, restore_root)
        model_class_name = restored_dict[MODEL_CLASS_NAME]

        recover_validation = None
        if RECOVER_VAL in restored_dict:
            recover_val_id = restored_dict[RECOVER_VAL]
            recover_validation = RecoverVal.load(recover_val_id, file_pers_service, dict_pers_service, restore_root)

        return cls(weights_file_path=weights_file_path, model_code_file_path=model_code_file_path,
                   model_class_name=model_class_name, store_id=store_id, recover_validation=recover_validation)

    def size_in_bytes(self, file_pers_service: AbstractFilePersistenceService,
                      dict_pers_service: AbstractDictPersistenceService) -> int:
        result = 0

        # size of the dict
        result += dict_pers_service.dict_size(self.store_id, RECOVER_INFO)

        restored_dict = dict_pers_service.recover_dict(self.store_id, RECOVER_INFO)
        # size of all referenced files/objects

        result += file_pers_service.file_size(restored_dict[WEIGHTS])
        result += file_pers_service.file_size(restored_dict[MODEL_CODE])
        if self.recover_validation:
            result += self.recover_validation.size_in_bytes(file_pers_service, dict_pers_service)

        return result

    def _representation_type(self) -> str:
        return RECOVER_INFO


class ProvenanceRecoverInfo(AbstractRecoverInfo):

    def __init__(self, dataset: Dataset, model_code_file_path, model_class_name: str, train_info: TrainInfo,
                 store_id: str = None, recover_validation: RecoverVal = None):
        super().__init__(store_id)
        self.dataset = dataset
        self.model_code_file_path = model_code_file_path
        self.model_class_name = model_class_name
        self.train_info = train_info
        self.recover_validation = recover_validation

    def persist(self, file_pers_service: AbstractFilePersistenceService,
                dict_pers_service: AbstractDictPersistenceService) -> str:

        if not self.store_id:
            self.store_id = dict_pers_service.generate_id()

        dataset_id = self.dataset.persist(file_pers_service, dict_pers_service)
        model_code_id = file_pers_service.save_file(self.model_code_file_path)
        train_info_id = self.train_info.persist(file_pers_service, dict_pers_service)

        dict_representation = {
            ID: self.store_id,
            DATASET: dataset_id,
            MODEL_CODE: model_code_id,
            MODEL_CLASS_NAME: self.model_class_name,
            TRAIN_INFO: train_info_id
        }

        if self.recover_validation:
            recover_val_id = self.recover_validation.persist(file_pers_service, dict_pers_service)
            dict_representation[RECOVER_VAL] = recover_val_id

        dict_pers_service.save_dict(dict_representation, RECOVER_INFO)

        return self.store_id

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService, restore_root: str):

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

        recover_validation = None
        if RECOVER_VAL in restored_dict:
            recover_val_id = restored_dict[RECOVER_VAL]
            recover_validation = RecoverVal.load(recover_val_id, file_pers_service, dict_pers_service, restore_root)

        return cls(dataset=dataset, model_class_name=model_class_name, train_info=train_info, store_id=store_id,
                   recover_validation=recover_validation, model_code_file_path=model_code_file_path)

    def size_in_bytes(self, file_pers_service: AbstractFilePersistenceService,
                      dict_pers_service: AbstractDictPersistenceService) -> int:
        result = 0

        # size of the dict
        result += dict_pers_service.dict_size(self.store_id, RECOVER_INFO)

        restored_dict = dict_pers_service.recover_dict(self.store_id, RECOVER_INFO)

        result += self.dataset.size_in_bytes(file_pers_service, dict_pers_service)
        result += file_pers_service.file_size(restored_dict[MODEL_CODE])
        result += self.train_info.size_in_bytes(file_pers_service, dict_pers_service)

        if self.recover_validation:
            result += self.recover_validation.size_in_bytes(file_pers_service, dict_pers_service)

        return result


def _data_dst_path():
    # TODO magic strings
    config_file = os.getenv('MMLIB_CONFIG')
    config = configparser.ConfigParser()
    config.read(config_file)

    return config['VALUES']['current_data_root']

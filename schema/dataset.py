import os

from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from schema.schema_obj import SchemaObj
from util.zip import zip_path, unzip

ID = 'id'
RAW_DATA = 'raw_data'

DATASET = 'dataset'


class Dataset(SchemaObj):

    def __init__(self, raw_data: str, store_id: str = None):
        super().__init__(store_id)
        self.raw_data = raw_data

    def persist(self, file_pers_service: AbstractFilePersistenceService,
                dict_pers_service: AbstractDictPersistenceService) -> str:
        super().persist(file_pers_service, dict_pers_service)

        zip_file_path = zip_path(self.raw_data)
        raw_data_id = file_pers_service.save_file(zip_file_path)

        dict_representation = {
            ID: self.store_id,
            RAW_DATA: raw_data_id
        }

        dict_pers_service.save_dict(dict_representation, DATASET)

        return self.store_id

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService, restore_root: str):
        restored_dict = dict_pers_service.recover_dict(obj_id, DATASET)

        store_id = restored_dict[ID]
        raw_data_id = restored_dict[RAW_DATA]

        zip_file_path = file_pers_service.recover_file(raw_data_id, restore_root)
        restore_path = os.path.join(restore_root, 'data')
        unzipped_path = unzip(zip_file_path, restore_path)

        return cls(raw_data=unzipped_path, store_id=store_id)

    def size_in_bytes(self, file_pers_service: AbstractFilePersistenceService,
                      dict_pers_service: AbstractDictPersistenceService) -> int:
        return dict_pers_service.dict_size(self.store_id, DATASET)

    def _representation_type(self) -> str:
        return DATASET

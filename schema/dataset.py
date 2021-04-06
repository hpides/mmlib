import os

from mmlib.persistence import FilePersistenceService, DictPersistenceService
from schema.schema_obj import SchemaObj
from util.zip import zip_path, unzip

RAW_DATA = 'raw_data'

DATASET = 'dataset'


class Dataset(SchemaObj):

    def __init__(self, raw_data: str = None, store_id: str = None):
        super().__init__(store_id)
        self.raw_data = raw_data

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        zip_file_path = zip_path(self.raw_data)
        raw_data_id = file_pers_service.save_file(zip_file_path)

        dict_representation[RAW_DATA] = raw_data_id

    def load_all_fields(self, file_pers_service: FilePersistenceService,
                        dict_pers_service: DictPersistenceService, restore_root: str,
                        load_recursive: bool = True, load_files: bool = True):
        restored_dict = dict_pers_service.recover_dict(self.store_id, DATASET)
        self.raw_data = _recover_data(file_pers_service, load_files, restore_root, restored_dict)

    def size_in_bytes(self, file_pers_service: FilePersistenceService,
                      dict_pers_service: DictPersistenceService) -> int:
        return dict_pers_service.dict_size(self.store_id, DATASET)

    def _representation_type(self) -> str:
        return DATASET


def _recover_data(file_pers_service, load_files, restore_root, restored_dict):
    raw_data_id = restored_dict[RAW_DATA]
    raw_data = None
    if load_files:
        zip_file_path = file_pers_service.recover_file(raw_data_id, restore_root)
        restore_path = os.path.join(restore_root, 'data')
        raw_data = unzip(zip_file_path, restore_path)
    return raw_data

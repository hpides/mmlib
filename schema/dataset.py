from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from schema.schema_obj import SchemaObj

ID = 'id'
RAW_DATA = 'raw_data'

DATASET = 'dataset'


class Dataset(SchemaObj):

    def __init__(self, raw_data: str, store_id: str = None):
        self.store_id = store_id
        self.raw_data = raw_data

    def persist(self, file_pers_service: AbstractFilePersistenceService,
                dict_pers_service: AbstractDictPersistenceService) -> str:
        if not self.store_id:
            self.store_id = dict_pers_service.generate_id()

        dict_representation = {
            ID: self.store_id,
            RAW_DATA: self.raw_data
        }

        dict_pers_service.save_dict(dict_representation, DATASET)

        return self.store_id

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService, restore_root: str):
        restored_dict = dict_pers_service.recover_dict(obj_id, DATASET)

        store_id = restored_dict[ID]
        raw_data = restored_dict[RAW_DATA]

        return cls(raw_data=raw_data, store_id=store_id)

    def size_in_bytes(self, file_pers_service: AbstractFilePersistenceService,
                      dict_pers_service: AbstractDictPersistenceService) -> int:

        return dict_pers_service.dict_size(self.store_id, DATASET)


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
        pass

    def size_in_bytes(self, file_pers_service: AbstractFilePersistenceService,
                      dict_pers_service: AbstractDictPersistenceService) -> int:
        pass

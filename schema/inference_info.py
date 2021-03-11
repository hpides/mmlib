from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from schema.RestorableObject import RestorableObject
from schema.environment import Environment
from schema.schema_obj import SchemaObj

ID = 'id'
DATA_LOADER = 'data_loader'
PRE_PROCESSOR = 'pre_processor'
ENVIRONMENT = 'environment'

INFERENCE_INFO = 'inference_info'


class InferenceInfo(SchemaObj):

    def __init__(self, dataloader: RestorableObject, pre_processor: RestorableObject, environment: Environment,
                 store_id: str = None):
        self.store_id = store_id
        self.dataloader = dataloader
        self.pre_processor = pre_processor
        self.environment = environment

    def persist(self, file_pers_service: AbstractFilePersistenceService,
                dict_pers_service: AbstractDictPersistenceService) -> str:

        if not self.store_id:
            self.store_id = dict_pers_service.generate_id()

        dataloader_id = self.dataloader.persist(file_pers_service, dict_pers_service)
        pre_processor_id = self.pre_processor.persist(file_pers_service, dict_pers_service)
        environment = self.environment.persist(file_pers_service, dict_pers_service)

        dict_representation = {
            ID: self.store_id,
            DATA_LOADER: dataloader_id,
            PRE_PROCESSOR: pre_processor_id,
            ENVIRONMENT: environment
        }

        dict_pers_service.save_dict(dict_representation, INFERENCE_INFO)

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService, restore_root: str):
        pass

    def size_in_bytes(self, file_pers_service: AbstractFilePersistenceService,
                      dict_pers_service: AbstractDictPersistenceService) -> int:
        pass

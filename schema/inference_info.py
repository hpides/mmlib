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
        environment_id = self.environment.persist(file_pers_service, dict_pers_service)

        dict_representation = {
            ID: self.store_id,
            DATA_LOADER: dataloader_id,
            PRE_PROCESSOR: pre_processor_id,
            ENVIRONMENT: environment_id
        }

        dict_pers_service.save_dict(dict_representation, INFERENCE_INFO)

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService, restore_root: str):
        restored_dict = dict_pers_service.recover_dict(obj_id, INFERENCE_INFO)

        store_id = restored_dict[ID]
        dataloader_id = restored_dict[DATA_LOADER]
        dataloader = RestorableObject.load(dataloader_id, file_pers_service, dict_pers_service, restore_root)
        pre_processor_id = restored_dict[PRE_PROCESSOR]
        pre_processor = RestorableObject.load(pre_processor_id, file_pers_service, dict_pers_service, restore_root)
        environment_id = restored_dict[ENVIRONMENT]
        environment = Environment.load(environment_id, file_pers_service, dict_pers_service, restore_root)

        return cls(dataloader=dataloader, pre_processor=pre_processor, environment=environment, store_id=store_id)

    def size_in_bytes(self, file_pers_service: AbstractFilePersistenceService,
                      dict_pers_service: AbstractDictPersistenceService) -> int:

        result = 0

        # size of the dict
        result += dict_pers_service.dict_size(self.store_id, INFERENCE_INFO)

        # size of all referenced files/objects

        result += self.dataloader.size_in_bytes(file_pers_service, dict_pers_service)
        result += self.pre_processor.size_in_bytes(file_pers_service, dict_pers_service)
        result += self.environment.size_in_bytes(file_pers_service, dict_pers_service)

        return result

from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from schema.environment import Environment
from schema.restorable_object import StateDictRestorableObjectWrapper
from schema.schema_obj import SchemaObj


ENVIRONMENT = 'environment'
INFERENCE_SERVICE = 'inference_service'

INFERENCE_INFO = 'inference_info'


# TODO check for subclassing approach for InferenceInfo and trainInfo
class InferenceInfo(SchemaObj):

    def __init__(self, inference_service: StateDictRestorableObjectWrapper, environment: Environment,
                 store_id: str = None):
        super().__init__(store_id)
        self.environment = environment
        self.inference_service = inference_service

    def persist(self, file_pers_service: AbstractFilePersistenceService,
                dict_pers_service: AbstractDictPersistenceService) -> str:
        super().persist(file_pers_service, dict_pers_service)

        env_id = self.environment.persist(file_pers_service, dict_pers_service)
        train_service_id = self.inference_service.persist(file_pers_service, dict_pers_service)

        dict_representation = {
            ID: self.store_id,
            INFERENCE_SERVICE: train_service_id,
            ENVIRONMENT: env_id,
        }

        dict_pers_service.save_dict(dict_representation, INFERENCE_INFO)

        return self.store_id

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService, restore_root: str, load_recursive: bool = False,
             load_files: bool = False):
        restored_dict = dict_pers_service.recover_dict(obj_id, INFERENCE_INFO)

        store_id = restored_dict[ID]

        env_id = restored_dict[ENVIRONMENT]
        env = Environment.load(env_id, file_pers_service, dict_pers_service, restore_root)

        inference_service_id = restored_dict[INFERENCE_SERVICE]
        inference_service_wrapper = StateDictRestorableObjectWrapper.load(inference_service_id, file_pers_service,
                                                                          dict_pers_service, restore_root)
        inference_service_wrapper.restore_instance()

        return cls(inference_service=inference_service_wrapper, environment=env, store_id=store_id)

    def size_in_bytes(self, file_pers_service: AbstractFilePersistenceService,
                      dict_pers_service: AbstractDictPersistenceService) -> int:
        result = 0

        result += dict_pers_service.dict_size(self.store_id, INFERENCE_INFO)

        result += self.environment.size_in_bytes(file_pers_service, dict_pers_service)
        result += self.inference_service.size_in_bytes(file_pers_service, dict_pers_service)

        return result

    def _representation_type(self) -> str:
        return INFERENCE_INFO

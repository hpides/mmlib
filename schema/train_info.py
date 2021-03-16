from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from schema.environment import Environment
from schema.restorable_object import StateDictRestorableObjectWrapper
from schema.schema_obj import SchemaObj

ID = 'id'
ENVIRONMENT = 'environment'
TRAIN_SERVICE = 'train_service'

TRAIN_INFO = 'train_info'


class TrainInfo(SchemaObj):

    def __init__(self, train_service: StateDictRestorableObjectWrapper, environment: Environment, store_id: str = None):
        self.store_id = store_id
        self.environment = environment
        self.train_service = train_service

    def persist(self, file_pers_service: AbstractFilePersistenceService,
                dict_pers_service: AbstractDictPersistenceService) -> str:
        if not self.store_id:
            self.store_id = dict_pers_service.generate_id()

        env_id = self.environment.persist(file_pers_service, dict_pers_service)
        train_service_id = self.train_service.persist(file_pers_service, dict_pers_service)

        dict_representation = {
            ID: self.store_id,
            TRAIN_SERVICE: train_service_id,
            ENVIRONMENT: env_id,
        }

        dict_pers_service.save_dict(dict_representation, TRAIN_INFO)

        return self.store_id

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService, restore_root: str):
        restored_dict = dict_pers_service.recover_dict(obj_id, TRAIN_INFO)

        store_id = restored_dict[ID]

        env_id = restored_dict[ENVIRONMENT]
        env = Environment.load(env_id, file_pers_service, dict_pers_service, restore_root)

        train_service_id = restored_dict[TRAIN_SERVICE]
        train_service_wrapper = StateDictRestorableObjectWrapper.load(train_service_id, file_pers_service,
                                                                      dict_pers_service, restore_root)
        train_service_wrapper.restore_instance()

        return cls(train_service=train_service_wrapper, environment=env, store_id=store_id)

    def size_in_bytes(self, file_pers_service: AbstractFilePersistenceService,
                      dict_pers_service: AbstractDictPersistenceService) -> int:
        result = 0

        result += dict_pers_service.dict_size(self.store_id, TRAIN_INFO)

        result += self.environment.size_in_bytes(file_pers_service, dict_pers_service)
        result += self.train_service.size_in_bytes(file_pers_service, dict_pers_service)

        return result

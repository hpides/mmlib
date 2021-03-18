from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from schema.environment import Environment
from schema.restorable_object import StateDictRestorableObjectWrapper
from schema.schema_obj import SchemaObj

ID = 'id'
TRAIN_SERVICE = 'train_service'
TRAIN_KWARGS = 'train_kwargs'
ENVIRONMENT = 'environment'

TRAIN_INFO = 'train_info'


class TrainInfo(SchemaObj):

    def __init__(self, train_service: StateDictRestorableObjectWrapper, train_kwargs: dict, environment: Environment,
                 store_id: str = None):
        self.store_id = store_id
        self.train_service = train_service
        self.train_kwargs = train_kwargs
        self.environment = environment

    def persist(self, file_pers_service: AbstractFilePersistenceService,
                dict_pers_service: AbstractDictPersistenceService) -> str:
        if not self.store_id:
            self.store_id = dict_pers_service.generate_id()

        env_id = self.environment.persist(file_pers_service, dict_pers_service)
        train_service_id = self.train_service.persist(file_pers_service, dict_pers_service)

        dict_representation = {
            ID: self.store_id,
            TRAIN_SERVICE: train_service_id,
            TRAIN_KWARGS: self.train_kwargs,
            ENVIRONMENT: env_id,
        }

        dict_pers_service.save_dict(dict_representation, TRAIN_INFO)

        return self.store_id

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService, restore_root: str):
        restored_dict = dict_pers_service.recover_dict(obj_id, TRAIN_INFO)

        store_id = restored_dict[ID]

        train_service_id = restored_dict[TRAIN_SERVICE]
        train_service_wrapper = StateDictRestorableObjectWrapper.load(train_service_id, file_pers_service,
                                                                      dict_pers_service, restore_root)
        train_service_wrapper.restore_instance(file_pers_service, dict_pers_service, restore_root)

        train_kwargs = restored_dict[TRAIN_KWARGS]

        env_id = restored_dict[ENVIRONMENT]
        env = Environment.load(env_id, file_pers_service, dict_pers_service, restore_root)

        return cls(train_service=train_service_wrapper, train_kwargs=train_kwargs, environment=env, store_id=store_id)

    def size_in_bytes(self, file_pers_service: AbstractFilePersistenceService,
                      dict_pers_service: AbstractDictPersistenceService) -> int:
        result = 0

        result += dict_pers_service.dict_size(self.store_id, TRAIN_INFO)

        result += self.train_service.size_in_bytes(file_pers_service, dict_pers_service)
        result += self.environment.size_in_bytes(file_pers_service, dict_pers_service)

        return result

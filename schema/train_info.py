from mmlib.constants import ID
from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from schema.environment import Environment
from schema.restorable_object import StateDictRestorableObjectWrapper
from schema.schema_obj import SchemaObj
from util.init_from_file import create_type


TRAIN_SERVICE = 'train_service'
TRAIN_KWARGS = 'train_kwargs'
WRAPPER_CODE = 'wrapper_code'
WRAPPER_CLASS_NAME = 'wrapper_class_name'
ENVIRONMENT = 'environment'

TRAIN_INFO = 'train_info'


class TrainInfo(SchemaObj):

    def load_all_fields(self, file_pers_service: AbstractFilePersistenceService,
                        dict_pers_service: AbstractDictPersistenceService, restore_root: str, load_ref_fields=True):
        pass

    @classmethod
    def load_placeholder(cls, obj_id: str):
        pass

    def __init__(self, ts_wrapper: StateDictRestorableObjectWrapper, ts_wrapper_code: str, ts_wrapper_class_name: str,
                 train_kwargs: dict, environment: Environment,
                 store_id: str = None):
        super().__init__(store_id)
        self.train_service_wrapper = ts_wrapper
        self.train_service_wrapper_code = ts_wrapper_code
        self.train_service_wrapper_class_name = ts_wrapper_class_name
        self.train_kwargs = train_kwargs
        self.environment = environment

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        env_id = self.environment.persist(file_pers_service, dict_pers_service)
        train_service_id = self.train_service_wrapper.persist(file_pers_service, dict_pers_service)

        print('train_service_ID')
        print(train_service_id)

        dict_representation[TRAIN_SERVICE] = train_service_id
        dict_representation[WRAPPER_CODE] = self.train_service_wrapper_code
        dict_representation[WRAPPER_CLASS_NAME] = self.train_service_wrapper_class_name
        dict_representation[TRAIN_KWARGS] = self.train_kwargs
        dict_representation[ENVIRONMENT] = env_id

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService, restore_root: str):
        restored_dict = dict_pers_service.recover_dict(obj_id, TRAIN_INFO)

        store_id = restored_dict[ID]

        train_service_id = restored_dict[TRAIN_SERVICE]

        ts_wrapper_class_name = restored_dict[WRAPPER_CLASS_NAME]
        ts_wrapper_code = restored_dict[WRAPPER_CODE]
        wrapper_class = create_type(code=ts_wrapper_code, type_name=ts_wrapper_class_name)

        train_service_wrapper = wrapper_class.load(train_service_id, file_pers_service,
                                                   dict_pers_service, restore_root)
        train_service_wrapper.restore_instance(file_pers_service, dict_pers_service, restore_root)

        train_kwargs = restored_dict[TRAIN_KWARGS]

        env_id = restored_dict[ENVIRONMENT]
        env = Environment.load(env_id, file_pers_service, dict_pers_service, restore_root)

        return cls(ts_wrapper=train_service_wrapper, ts_wrapper_code=ts_wrapper_code,
                   ts_wrapper_class_name=ts_wrapper_class_name, train_kwargs=train_kwargs, environment=env,
                   store_id=store_id)

    def size_in_bytes(self, file_pers_service: AbstractFilePersistenceService,
                      dict_pers_service: AbstractDictPersistenceService) -> int:
        result = 0

        result += dict_pers_service.dict_size(self.store_id, TRAIN_INFO)

        result += self.train_service_wrapper.size_in_bytes(file_pers_service, dict_pers_service)
        result += self.environment.size_in_bytes(file_pers_service, dict_pers_service)

        return result

    def _representation_type(self) -> str:
        return TRAIN_INFO

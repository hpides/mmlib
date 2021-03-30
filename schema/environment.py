from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from schema.schema_obj import SchemaObj

ID = 'id'
ENVIRONMENT_DICT = 'environment_dict'

ENVIRONMENT = 'environment'


class Environment(SchemaObj):

    def __init__(self, environment_data: dict, store_id: str = None):
        super().__init__(store_id)
        self.environment_data = environment_data

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        environment_data_id = dict_pers_service.save_dict(self.environment_data, ENVIRONMENT_DICT)

        dict_representation[ENVIRONMENT_DICT] = environment_data_id

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService, restore_root: str):
        restored_dict = dict_pers_service.recover_dict(obj_id, ENVIRONMENT)

        env_dict = dict_pers_service.recover_dict(restored_dict[ENVIRONMENT_DICT], ENVIRONMENT_DICT)

        return cls(store_id=obj_id, environment_data=env_dict)

    def size_in_bytes(self, file_pers_service: AbstractFilePersistenceService,
                      dict_pers_service: AbstractDictPersistenceService) -> int:
        restored_dict = dict_pers_service.recover_dict(self.store_id, ENVIRONMENT)
        env_size = dict_pers_service.dict_size(restored_dict[ENVIRONMENT_DICT], ENVIRONMENT_DICT)

        return dict_pers_service.dict_size(self.store_id, ENVIRONMENT) + env_size

    def _representation_type(self) -> str:
        return ENVIRONMENT

import abc

from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from schema.recover_val import RecoverVal
from schema.schema_obj import SchemaObj


class AbstractRecoverInfo(SchemaObj, metaclass=abc.ABCMeta):
    pass


ID = 'id'
WEIGHTS = 'weights'
MODEL_CODE = 'model_code'
CODE_NAME = 'code_name'
RECOVER_VAL = 'recover_val'

REPRESENT_TYPE = 'recover_info'


class FullModelRecoverInfo(AbstractRecoverInfo):

    def __init__(self, weights_file_path: str = None, model_code_file_path: str = None, model_class_name: str = None,
                 store_id: str = None, recover_validation: RecoverVal = None):
        self.store_id = store_id
        self.weights_file_path = weights_file_path
        self.model_code_file_path = model_code_file_path
        self.model_class_name = model_class_name
        self.recover_validation = recover_validation

    def persist(self, file_pers_service: AbstractFilePersistenceService,
                dict_pers_service: AbstractDictPersistenceService) -> str:

        if not self.store_id:
            self.store_id = dict_pers_service.generate_id()

        weights_id = file_pers_service.save_file(self.weights_file_path)
        model_code_id = file_pers_service.save_file(self.model_code_file_path)
        recover_val_id = self.recover_validation.persist(file_pers_service, dict_pers_service)

        dict_representation = {
            ID: self.store_id,
            WEIGHTS: weights_id,
            MODEL_CODE: model_code_id,
            CODE_NAME: recover_val_id,
        }

        if self.recover_validation:
            dict_representation[RECOVER_VAL] = recover_val_id

        dict_id = dict_pers_service.save_dict(dict_representation, REPRESENT_TYPE)
        assert dict_id == self.store_id, 'ids should be the same'

        return self.store_id

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService):
        pass

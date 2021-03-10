import abc

from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from schema.recover_val import RecoverVal
from schema.schema_obj import SchemaObj


class AbstractRecoverInfo(SchemaObj, metaclass=abc.ABCMeta):
    pass


ID = 'id'
WEIGHTS = 'weights'
MODEL_CODE = 'model_code'
MODEL_CLASS_NAME = 'model_class_name'
RECOVER_VAL = 'recover_val'

REPRESENT_TYPE = 'recover_info'


class FullModelRecoverInfo(AbstractRecoverInfo):


    def __init__(self, weights_file_path: str, model_code_file_path, model_class_name: str,
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
            MODEL_CLASS_NAME: recover_val_id,
        }

        if self.recover_validation:
            dict_representation[RECOVER_VAL] = recover_val_id

        dict_pers_service.save_dict(dict_representation, REPRESENT_TYPE)

        return self.store_id

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService, restore_root: str):

        restored_dict = dict_pers_service.recover_dict(obj_id, REPRESENT_TYPE)

        store_id = restored_dict[ID]
        weights_file_id = restored_dict[WEIGHTS]
        # TODO think about where to store files
        weights_file_path = file_pers_service.recover_file(weights_file_id, restore_root)
        model_code_file_id = restored_dict[MODEL_CODE]
        model_code_file_path = file_pers_service.recover_file(model_code_file_id, restore_root)
        model_class_name = restored_dict[MODEL_CLASS_NAME]

        recover_validation = None
        if RECOVER_VAL in restored_dict:
            recover_val_id = restored_dict[RECOVER_VAL]
            recover_validation = RecoverVal.load(recover_val_id, file_pers_service, dict_pers_service)

        return cls(weights_file_path=weights_file_path, model_code_file_path=model_code_file_path,
                   model_class_name=model_class_name, store_id=store_id, recover_validation=recover_validation)

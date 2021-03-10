from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from schema.schema_obj import SchemaObj

ID = 'id'
WEIGHTS_HASH = 'weights_hash'
INFERENCE_HASH = 'inference_hash'
DUMMY_INPUT_SHAPE = 'dummy_input_shape'

REPRESENT_TYPE = 'recover_val'


class RecoverVal(SchemaObj):

    def __init__(self, weights_hash: str, inference_hash:, dummy_input_shape: [int], store_id: str = None):
        self.store_id = store_id
        self.weights_hash = weights_hash
        self.inference_hash = inference_hash
        self.dummy_input_shape = dummy_input_shape

    def persist(self, file_pers_service: AbstractFilePersistenceService,
                dict_pers_service: AbstractDictPersistenceService) -> str:
        pass

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService):
        pass

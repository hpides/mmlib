from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from schema.schema_obj import SchemaObj

ID = 'id'
WEIGHTS_HASH = 'weights_hash'
INFERENCE_HASH = 'inference_hash'
DUMMY_INPUT_SHAPE = 'dummy_input_shape'

REPRESENT_TYPE = 'recover_val'


class RecoverVal(SchemaObj):

    def __init__(self, weights_hash: str, inference_hash: str, dummy_input_shape: [int], store_id: str = None):
        self.store_id = store_id
        self.weights_hash = weights_hash
        self.inference_hash = inference_hash
        self.dummy_input_shape = dummy_input_shape

    def persist(self, file_pers_service: AbstractFilePersistenceService,
                dict_pers_service: AbstractDictPersistenceService) -> str:
        if not self.store_id:
            self.store_id = dict_pers_service.generate_id()

        dict_representation = {
            ID: self.store_id,
            WEIGHTS_HASH: self.weights_hash,
            INFERENCE_HASH: self.inference_hash,
            DUMMY_INPUT_SHAPE: self.dummy_input_shape,
        }

        dict_pers_service.save_dict(dict_representation, REPRESENT_TYPE)

        return self.store_id

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService, restore_root: str):
        restored_dict = dict_pers_service.recover_dict(obj_id, REPRESENT_TYPE)
        store_id = restored_dict[ID]
        weights_hash = restored_dict[WEIGHTS_HASH]
        inference_hash = restored_dict[INFERENCE_HASH]
        dummy_input_shape = restored_dict[DUMMY_INPUT_SHAPE]

        return cls(weights_hash=weights_hash, inference_hash=inference_hash, dummy_input_shape=dummy_input_shape,
                   store_id=store_id)

    def size_in_bytes(self, file_pers_service: AbstractFilePersistenceService,
                      dict_pers_service: AbstractDictPersistenceService) -> int:
        return dict_pers_service.dict_size(self.store_id, REPRESENT_TYPE)

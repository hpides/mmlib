from schema.function import Function
from schema.schema_obj import SchemaObj, SchemaObjType

ID = 'id'
WEIGHTS_HASH = 'weights_hash'
INFERENCE_HASH = 'inference_hash'
DUMMY_INPUT_SHAPE = 'dummy_input_shape'


class RecoverVal(SchemaObj):

    def __init__(self, r_id: str = None, weights_hash: str = None, inference_hash: str = None,
                 dummy_input_shape: [int] = None):
        self.r_id = r_id
        self.weights_hash = weights_hash
        self.inference_hash = inference_hash
        self.dummy_input_shape = dummy_input_shape
        self._type_mapping = {
            ID: SchemaObjType.STRING,
            WEIGHTS_HASH: SchemaObjType.STRING,
            INFERENCE_HASH: SchemaObjType.STRING,
            DUMMY_INPUT_SHAPE: SchemaObjType.STRING,
        }

    def to_dict(self) -> dict:
        recover_val = {
            WEIGHTS_HASH: self.weights_hash,
            INFERENCE_HASH: self.inference_hash,
            DUMMY_INPUT_SHAPE: self.dummy_input_shape,
        }

        if self.r_id:
            recover_val[ID] = self.r_id

        return recover_val

    def load_dict(self, state_dict: dict):
        self.r_id = state_dict[ID] if ID in state_dict else None
        self.weights_hash = state_dict[WEIGHTS_HASH]
        self.inference_hash = state_dict[INFERENCE_HASH]
        self.dummy_input_shape = state_dict[DUMMY_INPUT_SHAPE]

    def get_type(self, dict_key) -> SchemaObjType:
        return self._type_mapping[dict_key]

from schema.schema_obj import SchemaObj, SchemaObjType

ID = 'id'
WEIGHTS_HASH = 'weights_hash'
INFERENCE_HASH = 'inference_hash'
INFERENCE_DATA = 'inference_data'


class RecoverVal(SchemaObj):

    def __init__(self, r_id: str, weights_hash: str, inference_hash: str, inference_data: str):
        self.r_id = r_id
        self.weights_hash = weights_hash
        self.inference_hash = inference_hash
        self.inference_data = inference_data

    def to_dict(self) -> dict:
        recover_val = {
            WEIGHTS_HASH: self.weights_hash,
            INFERENCE_HASH: self.inference_hash,
            INFERENCE_DATA: self.inference_data,
        }

        if self.r_id:
            recover_val[ID] = self.r_id

        return recover_val

    def load_dict(self, state_dict: dict):
        self.r_id = state_dict[ID] if ID in state_dict else None
        self.weights_hash = state_dict[WEIGHTS_HASH]
        self.inference_hash = state_dict[INFERENCE_HASH]
        self.inference_data = state_dict[INFERENCE_DATA]


    def get_type(self, dict_key) -> SchemaObjType:
        pass

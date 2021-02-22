from schema.schema_obj import SchemaObj, SchemaObjType


class RecoverVal(SchemaObj):

    def __init__(self, r_id: str, weights_hash: str, inference_hash: str, inference_data: str):
        self.r_id = r_id
        self.weights_hash = weights_hash
        self.inference_hash = inference_hash
        self.inference_data = inference_data

    def to_dict(self) -> dict:
        pass

    def load_dict(self, state_dict: dict):
        pass

    def get_type(self, dict_key) -> SchemaObjType:
        pass

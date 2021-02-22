from schema.schema_obj import SchemaObj, SchemaObjType


class RecoverVal(SchemaObj):
    def to_dict(self) -> dict:
        pass

    def load_dict(self, state_dict: dict):
        pass

    def get_type(self, dict_key) -> SchemaObjType:
        pass
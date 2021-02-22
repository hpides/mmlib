from schema.schema_obj import SchemaObj, SchemaObjType

ID = 'id'
DATASET_CODE = 'dataset_code'
GENERATE_CALL = 'generate_call'
RAW_DATA = 'raw_data'
RAW_DATA_SIZE = 'raw_data_size'


class Dataset(SchemaObj):
    def to_dict(self) -> dict:
        pass

    def load_dict(self, state_dict: dict):
        pass

    def get_type(self, dict_key) -> SchemaObjType:
        pass

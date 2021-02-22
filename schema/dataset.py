from schema.schema_obj import SchemaObj, SchemaObjType

ID = 'id'
DATASET_CODE = 'dataset_code'
GENERATE_CALL = 'generate_call'
RAW_DATA = 'raw_data'
RAW_DATA_SIZE = 'raw_data_size'


class Dataset(SchemaObj):

    def __init__(self, d_id: str, dataset_code: str, generate_call: str, raw_data: str, raw_data_size: str):
        self.d_id = d_id
        self.dataset_code = dataset_code
        self.generate_call = generate_call
        self.raw_data = raw_data
        self.raw_data_size = raw_data_size
        self.type_mapping = {
            ID: SchemaObjType.STRING,
            DATASET_CODE: SchemaObjType.FILE,
            GENERATE_CALL: SchemaObjType.STRING,
            RAW_DATA: SchemaObjType.FILE,
            RAW_DATA_SIZE: SchemaObjType.STRING,
        }

    def to_dict(self) -> dict:
        pass

    def load_dict(self, state_dict: dict):
        pass

    def get_type(self, dict_key) -> SchemaObjType:
        pass

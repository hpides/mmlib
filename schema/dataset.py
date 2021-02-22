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
        self._type_mapping = {
            ID: SchemaObjType.STRING,
            DATASET_CODE: SchemaObjType.FILE,
            GENERATE_CALL: SchemaObjType.STRING,
            RAW_DATA: SchemaObjType.FILE,
            RAW_DATA_SIZE: SchemaObjType.STRING,
        }

    def to_dict(self) -> dict:
        dataset = {
            DATASET_CODE: self.dataset_code,
            GENERATE_CALL: self.generate_call,
            RAW_DATA: self.raw_data,
            RAW_DATA_SIZE: self.raw_data_size,
        }

        if self.d_id:
            dataset[ID] = self.d_id

        return dataset

    def load_dict(self, state_dict: dict):
        self.d_id = state_dict[ID] if ID in state_dict else None
        self.dataset_code = state_dict[DATASET_CODE]
        self.generate_call = state_dict[GENERATE_CALL]
        self.raw_data = state_dict[RAW_DATA]
        self.raw_data_size = state_dict[RAW_DATA_SIZE]

    def get_type(self, dict_key) -> SchemaObjType:
        return self._type_mapping[dict_key]

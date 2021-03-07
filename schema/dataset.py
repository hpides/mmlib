from schema.schema_obj import SchemaObj, SchemaObjType

ID = 'id'
DATASET_CODE = 'dataset_code'
CLASS_NAME = 'class_name'
RAW_DATA = 'raw_data'
RAW_DATA_SIZE = 'raw_data_size'


class Dataset(SchemaObj):

    def __init__(self, d_id: str = None, dataset_code: str = None, class_name: str = None, raw_data: str = None,
                 raw_data_size: str = None):
        self.d_id = d_id
        self.dataset_code = dataset_code
        self.class_name = class_name
        self.raw_data = raw_data
        self.raw_data_size = raw_data_size
        self._type_mapping = {
            ID: SchemaObjType.STRING,
            DATASET_CODE: SchemaObjType.FILE,
            CLASS_NAME: SchemaObjType.STRING,
            RAW_DATA: SchemaObjType.FILE,
            RAW_DATA_SIZE: SchemaObjType.STRING,
        }

    def to_dict(self) -> dict:
        dataset = {
            DATASET_CODE: self.dataset_code,
            CLASS_NAME: self.class_name,
            RAW_DATA: self.raw_data,
            RAW_DATA_SIZE: self.raw_data_size,
        }

        if self.d_id:
            dataset[ID] = self.d_id

        return dataset

    def load_dict(self, state_dict: dict):
        self.d_id = state_dict[ID] if ID in state_dict else None
        self.dataset_code = state_dict[DATASET_CODE]
        self.class_name = state_dict[CLASS_NAME]
        self.raw_data = state_dict[RAW_DATA]
        self.raw_data_size = state_dict[RAW_DATA_SIZE]

    def get_type(self, dict_key) -> SchemaObjType:
        return self._type_mapping[dict_key]

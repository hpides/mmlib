from schema.schema_obj import SchemaObj, SchemaObjType

ID = 'id'
DATA_LOADER = 'data_loader'
PRE_PROCESSOR = 'pre_processor'
ENVIRONMENT = 'environment'


class InferenceInfo(SchemaObj):

    def __init__(self, i_id: str = None, dataloader: str = None, pre_processor: str = None, environment: str = None):
        self.i_id = i_id
        self.dataloader = dataloader
        self.pre_processor = pre_processor
        self.environment = environment
        self._type_mapping = {
            ID: SchemaObjType.STRING,
            DATA_LOADER: SchemaObjType.RESTORABLE_OBJ,
            PRE_PROCESSOR: SchemaObjType.RESTORABLE_OBJ,
            ENVIRONMENT: SchemaObjType.ENVIRONMENT,
        }

    def to_dict(self) -> dict:
        inference_info = {
            DATA_LOADER: self.dataloader,
            PRE_PROCESSOR: self.pre_processor,
            ENVIRONMENT: self.environment,
        }

        if self.i_id:
            inference_info[ID] = self.i_id

        return inference_info

    def load_dict(self, state_dict: dict):
        self.i_id = state_dict[ID] if ID in state_dict else None
        self.dataloader = state_dict[DATA_LOADER]
        self.pre_processor = state_dict[PRE_PROCESSOR]
        self.environment = state_dict[ENVIRONMENT]

    def get_type(self, dict_key) -> SchemaObjType:
        return self._type_mapping[dict_key]

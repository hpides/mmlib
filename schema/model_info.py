from schema.schema_obj import SchemaObj, SchemaObjType

ID = 'id'
STORE_TYPE = 'store_type'
RECOVER_INFO = 'recover_info'
DERIVED_FROM = 'derived_from'
INFERENCE_INFO = 'inference_info'
TRAIN_INFO = 'train_info'


class ModelInfo(SchemaObj):

    def __init__(self, m_id: str = None, store_type: str = None, recover_info: str = None, derived_from: str = None,
                 inference_info: str = None, train_info: str = None):
        self.m_id = m_id
        self.store_type = store_type
        self.recover_info = recover_info
        self.derived_from = derived_from
        self.inference_info = inference_info
        self.train_info = train_info
        self._type_mapping = {
            ID: SchemaObjType.STRING,
            STORE_TYPE: SchemaObjType.STRING,
            DERIVED_FROM: SchemaObjType.MODEL_INFO,
            INFERENCE_INFO: SchemaObjType.STRING,  # TODO to specify
            TRAIN_INFO: SchemaObjType.STRING,  # TODO to specify
        }

    def load_dict(self, state_dict):
        self.m_id = state_dict[ID] if ID in state_dict else None
        self.store_type = state_dict[STORE_TYPE]
        self.recover_info = state_dict[RECOVER_INFO]
        self.derived_from = state_dict[DERIVED_FROM] if DERIVED_FROM in state_dict else None
        self.inference_info = state_dict[INFERENCE_INFO] if INFERENCE_INFO in state_dict else None
        self.train_info = state_dict[TRAIN_INFO] if TRAIN_INFO in state_dict else None

    def to_dict(self):
        model_info = {
            STORE_TYPE: self.store_type,
            RECOVER_INFO: self.recover_info,
        }

        if self.m_id:
            model_info[ID] = self.m_id
        if self.derived_from:
            model_info[DERIVED_FROM] = self.derived_from
        if self.inference_info:
            model_info[INFERENCE_INFO] = self.derived_from
        if self.train_info:
            model_info[TRAIN_INFO] = self.derived_from

        return model_info

    def get_type(self, dict_key) -> SchemaObjType:
        if dict_key == RECOVER_INFO:
            if self.store_type == '1':
                return SchemaObjType.RECOVER_T1
            else:
                assert False, 'not implemented yet'
        else:
            return self._type_mapping[dict_key]

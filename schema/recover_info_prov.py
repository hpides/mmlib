from schema.schema_obj import SchemaObj, SchemaObjType

ID = 'id'
DATASET = 'dataset'
MODEL_CODE = 'model_code'
CODE_NAME = 'code_name'
TRAIN_INFO = 'train_info'
RECOVER_VAL = 'recover_val'


class RecoverInfoProv(SchemaObj):

    def __init__(self, r_id: str = None, dataset: str = None, model_code: str = None, code_name: str = None,
                 train_info: str = None, recover_validation: str = None):
        self.r_id = r_id
        self.dataset = dataset
        self.model_code = model_code
        self.code_name = code_name
        self.train_info = train_info
        self.recover_validation = recover_validation
        self._type_mapping = {
            ID: SchemaObjType.STRING,
            DATASET: SchemaObjType.DATASET,
            MODEL_CODE: SchemaObjType.FILE,
            CODE_NAME: SchemaObjType.STRING,
            TRAIN_INFO: SchemaObjType.TRAIN_INFO,
            RECOVER_VAL: SchemaObjType.RECOVER_VAL
        }

    def load_dict(self, state_dict):
        # mandatory fields
        self.r_id = state_dict[ID]
        self.dataset = state_dict[DATASET]
        # (maybe) optional
        self.train_info = state_dict[TRAIN_INFO] if TRAIN_INFO in state_dict else None
        self.recover_validation = state_dict[RECOVER_VAL] if RECOVER_VAL in state_dict else None

        # the fields model_code and code_name are fields that should be inferred/merged from the base model

    def infer_and_merge(self, base_info_dict):
        # inferred / merged
        self.model_code = base_info_dict[MODEL_CODE]
        self.code_name = base_info_dict[CODE_NAME]
        self.train_info = base_info_dict[TRAIN_INFO]

    def to_dict(self, include_inferred=False):
        recover_info_prov = {
            ID: self.r_id,
            DATASET: self.dataset,
        }

        if self.r_id:
            recover_info_prov[ID] = self.r_id
        if self.recover_validation:
            recover_info_prov[RECOVER_VAL] = self.recover_validation

        if include_inferred:
            if self.model_code:
                recover_info_prov[MODEL_CODE] = self.model_code
            if self.code_name:
                recover_info_prov[CODE_NAME] = self.code_name
            if self.train_info:
                recover_info_prov[TRAIN_INFO] = self.train_info

        return recover_info_prov

    def get_type(self, dict_key) -> SchemaObjType:
        return self._type_mapping[dict_key]

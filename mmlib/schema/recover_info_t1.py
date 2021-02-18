ID = 'id'
WEIGHTS = 'weights'
MODEL_CODE = 'model_code'
IMPORT_ROOT = 'import_root'
CODE_NAME = 'code_name'
RECOVER_VAL = 'recover_val'


class RecoverInfoT1:

    def __init__(self, r_id: str = None, weights: str = None, model_code: str = None, code_name: str = None,
                 recover_validation: str = None):
        self.r_id = r_id
        self.weights = weights
        self.model_code = model_code
        self.code_name = code_name
        self.recover_validation = recover_validation

    def load_dict(self, state_dict):
        self.r_id = state_dict[ID]
        self.weights = state_dict[WEIGHTS]
        self.model_code = state_dict[MODEL_CODE]
        self.code_name = state_dict[CODE_NAME]
        self.recover_validation = state_dict[RECOVER_VAL] if RECOVER_VAL in state_dict else None

    def to_dict(self):
        recover_info_t1 = {
            ID: self.r_id,
            WEIGHTS: self.weights,
            MODEL_CODE: self.model_code,
            CODE_NAME: self.code_name,
            RECOVER_VAL: self.recover_validation  # TODO needs implementation
        }

        if self.r_id:
            recover_info_t1[ID] = self.r_id
        if self.recover_validation:
            recover_info_t1[RECOVER_VAL] = self.recover_validation

        return recover_info_t1

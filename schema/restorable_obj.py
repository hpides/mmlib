from schema.schema_obj import SchemaObj, SchemaObjType

ID = 'id'
CODE = 'code'
CLASS_NAME = 'class_name'
STATE = 'state'


class RestorableObj(SchemaObj):

    def __init__(self, r_id: str = None, code: str = None, class_name: str = None, state: str = None):
        self.r_id = r_id
        self.code = code
        self.class_name = class_name
        self.state = state
        self._type_mapping = {
            ID: SchemaObjType.STRING,
            CODE: SchemaObjType.FILE,
            CLASS_NAME: SchemaObjType.STRING,
            STATE: SchemaObjType.STRING,
        }

    def to_dict(self) -> dict:
        restorable_obj = {
            CODE: self.code,
            CLASS_NAME: self.class_name,
            STATE: self.state,
        }

        if self.r_id:
            restorable_obj[ID] = self.r_id

        return restorable_obj

    def load_dict(self, state_dict: dict):
        self.r_id = state_dict[ID] if ID in state_dict else None
        self.code = state_dict[CODE]
        self.class_name = state_dict[CLASS_NAME]
        self.state = state_dict[STATE]

    def get_type(self, dict_key) -> SchemaObjType:
        return self._type_mapping[dict_key]

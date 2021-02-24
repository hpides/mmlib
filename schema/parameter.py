from schema.schema_obj import SchemaObj, SchemaObjType

NAME = 'name'
PARAM_TYPE = 'param_type'
VALUE = 'value'
DEFAULT = 'default'


class Parameter(SchemaObj):
    def __init__(self, name: str, param_type: str, value: str, default: str):
        self.name = name
        self.param_type = param_type
        self.value = value
        self.default = default
        self._type_mapping = {
            NAME: SchemaObjType.STRING,
            PARAM_TYPE: SchemaObjType.STRING,
            VALUE: SchemaObjType.STRING,
            DEFAULT: SchemaObjType.STRING,
        }

    def to_dict(self) -> dict:
        parameter = {
            NAME: self.name,
            PARAM_TYPE: self.param_type,
            VALUE: self.value,
            DEFAULT: self.default,
        }

        return parameter

    def load_dict(self, state_dict: dict):
        self.name = state_dict[NAME]
        self.param_type = state_dict[PARAM_TYPE]
        self.value = state_dict[VALUE]
        self.default = state_dict[DEFAULT]

    def get_type(self, dict_key) -> SchemaObjType:
        return self._type_mapping[dict_key]

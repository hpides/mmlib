from schema.parameter import Parameter
from schema.schema_obj import SchemaObj, SchemaObjType

ID = 'id'
FUNCTION_CODE = 'function_code'
CALL_NAME = 'call_name'
FUNCTION_ARGS = 'function_args'
FUNCTION_KWARGS = 'function_kwargs'


class Function(SchemaObj):

    def __init__(self, f_id: str = None, function_code: str = None, call_name: str = None,
                 function_args: [Parameter] = None, function_kwargs: [Parameter] = None):
        self.f_id = f_id
        self.function_code = function_code
        self.call_name = call_name
        self.function_args = function_args
        self.function_kwargs = function_kwargs
        self._type_mapping = {
            ID: SchemaObjType.STRING,
            FUNCTION_CODE: SchemaObjType.FILE,
            CALL_NAME: SchemaObjType.STRING,
            FUNCTION_ARGS: SchemaObjType.PARAM_LIST,
            FUNCTION_KWARGS: SchemaObjType.PARAM_LIST,
        }

    def to_dict(self) -> dict:
        func_args

        function = {
            FUNCTION_CODE: self.function_code,
            CALL_NAME: self.call_name,
            FUNCTION_ARGS: self.function_args,
            FUNCTION_KWARGS: self.function_kwargs,
        }

        if self.f_id:
            function[ID] = self.f_id

        return function

    def load_dict(self, state_dict: dict):
        self.f_id = state_dict[ID] if ID in state_dict else None
        self.function_code = state_dict[FUNCTION_CODE]
        self.call_name = state_dict[CALL_NAME]
        self.function_args = state_dict[FUNCTION_ARGS]
        self.function_kwargs = state_dict[FUNCTION_KWARGS]

    def get_type(self, dict_key) -> SchemaObjType:
        return self._type_mapping[dict_key]

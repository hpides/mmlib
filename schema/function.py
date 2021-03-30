from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from schema.schema_obj import SchemaObj


CODE = 'code'
IMPORT_CMD = 'import_cmd'
CALL_NAME = 'call_name'
ARGS = 'args'
REF_TYPE_ARGS = 'ref_type_args'

FUNCTION = 'function'


class Function(SchemaObj):

    def __init__(self, code_file: str, import_cmd: str, call_name: str, args: dict, ref_type_args: [str],
                 store_id: str = None):
        super().__init__(store_id)
        self.code = code_file
        self.import_cmd = import_cmd
        self.call_name = call_name
        self.args = args
        self.ref_type_args = ref_type_args

    def persist(self, file_pers_service: AbstractFilePersistenceService,
                dict_pers_service: AbstractDictPersistenceService) -> str:

        super().persist(file_pers_service, dict_pers_service)

        dict_representation = {
            ID: self.store_id,
            CALL_NAME: self.call_name,
            ARGS: self.args,
            REF_TYPE_ARGS: self.ref_type_args
        }

        if self.code:
            assert self.import_cmd is None, 'if code is set then there should be no import cmd'
            code_file_id = file_pers_service.save_file(self.code)
            dict_representation[CODE] = code_file_id

        if self.import_cmd:
            assert self.code is None, 'if import_cmd is set then there should be no code'
            dict_representation[IMPORT_CMD] = self.import_cmd

        dict_pers_service.save_dict(dict_representation, FUNCTION)

        return self.store_id

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService, restore_root: str):
        pass

    def size_in_bytes(self, file_pers_service: AbstractFilePersistenceService,
                      dict_pers_service: AbstractDictPersistenceService) -> int:
        pass

    def _representation_type(self) -> str:
        return FUNCTION

from mmlib.persistence import FilePersistenceService, DictPersistenceService
from schema.schema_obj import SchemaObj


class WeightsUpdate(SchemaObj):
    # TODO
    def load_all_fields(self, file_pers_service: FilePersistenceService, dict_pers_service: DictPersistenceService,
                        restore_root: str, load_recursive: bool = True, load_files: bool = True):
        pass

    def size_in_bytes(self, file_pers_service: FilePersistenceService,
                      dict_pers_service: DictPersistenceService) -> int:
        pass

    def _representation_type(self) -> str:
        pass

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        pass
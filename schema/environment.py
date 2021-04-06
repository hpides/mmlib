from mmlib.persistence import FilePersistenceService, DictPersistenceService
from schema.schema_obj import SchemaObj

ENVIRONMENT = 'environment'


class Environment(SchemaObj):

    def __init__(self, store_id: str = None, python_version: str = None, pytorch_version: str = None,
                 processor_info: str = None, gpu_types: [str] = None, pytorch_info: str = None,
                 python_platform_info: dict = None, pip_freeze: list = None):
        super().__init__(store_id)
        self.python_version = python_version
        self.pytorch_version = pytorch_version
        self.processor_info = processor_info
        self.gpu_types = gpu_types
        self.pytorch_info = pytorch_info
        self.python_platform_info = python_platform_info
        self.pip_freeze = pip_freeze

    def load_all_fields(self, file_pers_service: FilePersistenceService, dict_pers_service: DictPersistenceService,
                        restore_root: str, load_recursive: bool = True, load_files: bool = True):
        pass

    def size_in_bytes(self, file_pers_service: FilePersistenceService,
                      dict_pers_service: DictPersistenceService) -> int:
        raise NotImplementedError

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        dict_representation['python_version'] = self.python_version
        dict_representation['pytorch_version'] = self.python_version
        dict_representation['processor_info'] = self.processor_info
        dict_representation['gpu_types'] = self.gpu_types
        dict_representation['pytorch_info'] = self.pytorch_info
        dict_representation['python_platform_info'] = self.python_platform_info
        dict_representation['pytorch_version'] = self.pip_freeze

    def _representation_type(self) -> str:
        return ENVIRONMENT

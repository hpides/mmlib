from mmlib.persistence import FilePersistenceService, DictPersistenceService
from schema.schema_obj import SchemaObj

PYTHON_VERSION = 'python_version'
PYTORCH_VERSION = 'pytorch_version'
PROCESSOR_INFO = 'processor_info'
GPU_TYPES = 'gpu_types'
PYTORCH_INFO = 'pytorch_info'
PYTHON_PLATFORM_INFO = 'python_platform_info'
PIP_FREEZE = 'pip_freeze'

ENVIRONMENT = 'environment'


class Environment(SchemaObj):

    def __init__(self, store_id: str = None, python_version: str = None, pytorch_version: str = None,
                 processor_info: str = None, gpu_types: str = None, pytorch_info: str = None,
                 python_platform_info: str = None, pip_freeze: list = None):
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
        restored_dict = dict_pers_service.recover_dict(self.store_id, ENVIRONMENT)

        self.python_version = restored_dict[PYTHON_VERSION]
        self.pytorch_version = restored_dict[PYTORCH_VERSION]
        self.processor_info = restored_dict[PROCESSOR_INFO]
        self.gpu_types = restored_dict[GPU_TYPES]
        self.pytorch_info = restored_dict[PYTORCH_INFO]
        self.python_platform_info = restored_dict[PYTHON_PLATFORM_INFO]
        self.pip_freeze = restored_dict[PIP_FREEZE]

    def size_in_bytes(self, file_pers_service: FilePersistenceService,
                      dict_pers_service: DictPersistenceService) -> int:
        raise NotImplementedError

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        dict_representation[PYTHON_VERSION] = self.python_version
        dict_representation[PYTORCH_VERSION] = self.pytorch_version
        dict_representation[PROCESSOR_INFO] = self.processor_info
        dict_representation[GPU_TYPES] = self.gpu_types
        dict_representation[PYTORCH_INFO] = self.pytorch_info
        dict_representation[PYTHON_PLATFORM_INFO] = self.python_platform_info
        dict_representation[PIP_FREEZE] = self.pip_freeze

    def _representation_type(self) -> str:
        return ENVIRONMENT

    def _size_class_specific_fields(self, restored_dict, file_pers_service, dict_pers_service):
        # this class only holds meta information
        return 0

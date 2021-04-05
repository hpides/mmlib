from schema.schema_obj import SchemaObj

ENVIRONMENT_DICT = 'environment_dict'

ENVIRONMENT = 'environment'


class Environment(SchemaObj):

    def __init__(self, store_id: str = None, python_version: str = None, pytorch_version: str = None,
                 processor: str = None, gpu_types: [str] = None, cuda_version: str = None, cudnn_version: str = None,
                 driver_version: str = None, pytorch_info: str = None, python_platform_info: dict = None,
                 pip_freeze: list = None):
        super().__init__(store_id)
        self.python_version = python_version
        self.pytorch_version = pytorch_version
        self.processor = processor
        self.gpu_types = gpu_types
        self.cuda_version = cuda_version
        self.cudnn_version = cudnn_version
        self.driver_version = driver_version
        self.pytorch_info = pytorch_info
        self.python_platform_info = python_platform_info
        self.pip_freeze = pip_freeze

    def _representation_type(self) -> str:
        return ENVIRONMENT

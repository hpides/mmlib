from schema.schema_obj import SchemaObj

ID = 'id'
DATA_LOADER = 'data_loader'
PRE_PROCESSOR = 'pre_processor'
ENVIRONMENT = 'environment'


class InferenceInfo(SchemaObj):

    def __init__(self, i_id: str = None, dataloader: str = None, pre_processor: str = None, environment: str = None):
        self.i_id = i_id
        self.dataloader = dataloader
        self.pre_processor = pre_processor
        self.environment = environment

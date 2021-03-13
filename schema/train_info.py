from schema.schema_obj import SchemaObj

ID = 'id'
DATA_LOADER = 'data_loader'
PRE_PROCESSOR = 'pre_processor'
DATASET = 'dataset'
LOSS = 'loss'
OPTIMIZER = 'optimizer'
ENVIRONMENT = 'environment'


class TrainInfo(SchemaObj):

    def __init__(self, t_id: str = None, dataloader: str = None, pre_processor: str = None, dataset: str = None,
                 loss: str = None, optimizer: str = None, environment: str = None):
        self.t_id = t_id
        self.dataloader = dataloader
        self.pre_processor = pre_processor
        self.dataset = dataset
        self.loss = loss
        self.optimizer = optimizer
        self.environment = environment

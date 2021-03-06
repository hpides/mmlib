from schema.schema_obj import SchemaObj, SchemaObjType

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
        self._type_mapping = {
            ID: SchemaObjType.STRING,
            DATA_LOADER: SchemaObjType.RESTORABLE_OBJ,
            PRE_PROCESSOR: SchemaObjType.RESTORABLE_OBJ,
            DATASET: SchemaObjType.DATASET,
            LOSS: SchemaObjType.FUNCTION,
            OPTIMIZER: SchemaObjType.RESTORABLE_OBJ,
            ENVIRONMENT: SchemaObjType.ENVIRONMENT,
        }

    def to_dict(self) -> dict:
        inference_info = {
            DATA_LOADER: self.dataloader,
            PRE_PROCESSOR: self.pre_processor,
            DATASET: self.dataset,
            LOSS: self.loss,
            OPTIMIZER: self.optimizer,
            ENVIRONMENT: self.environment,
        }

        if self.t_id:
            inference_info[ID] = self.t_id

        return inference_info

    def load_dict(self, state_dict: dict):
        self.t_id = state_dict[ID] if ID in state_dict else None
        self.dataloader = state_dict[DATA_LOADER]
        self.pre_processor = state_dict[PRE_PROCESSOR]
        self.dataset = state_dict[DATASET]
        self.loss = state_dict[LOSS]
        self.optimizer = state_dict[OPTIMIZER]
        self.environment = state_dict[ENVIRONMENT]

    def get_type(self, dict_key) -> SchemaObjType:
        return self._type_mapping[dict_key]

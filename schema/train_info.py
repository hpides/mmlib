from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from schema.environment import Environment
from schema.function import Function
from schema.inference_info import InferenceInfo, DATA_WRAPPER, PRE_PROCESSOR, DATA_LOADER, ENVIRONMENT, ID
from schema.restorable_object import RestorableObjectWrapper

LOSS = 'loss'
OPTIMIZER = 'optimizer'

TRAIN_INFO = 'train_info'


class TrainInfo(InferenceInfo):

    def __init__(self, data_wrapper: RestorableObjectWrapper, dataloader: RestorableObjectWrapper,
                 pre_processor: RestorableObjectWrapper, environment: Environment, loss: Function,
                 optimizer: RestorableObjectWrapper, store_id: str = None):
        super().__init__(data_wrapper, dataloader, pre_processor, environment, store_id)
        self.loss = loss
        self.optimizer = optimizer

    def persist(self, file_pers_service: AbstractFilePersistenceService,
                dict_pers_service: AbstractDictPersistenceService) -> str:
        dict_representation = self._persist_fields(dict_pers_service, file_pers_service)

        dict_pers_service.save_dict(dict_representation, TRAIN_INFO)

        return self.store_id

    def _persist_fields(self, dict_pers_service, file_pers_service):
        dict_representation = super()._persist_fields(dict_pers_service, file_pers_service)

        loss_func_id = self.loss.persist(file_pers_service, dict_pers_service)
        optimizer_id = self.optimizer.persist(file_pers_service, dict_pers_service)

        dict_representation[LOSS] = loss_func_id
        dict_representation[OPTIMIZER] = optimizer_id

        return dict_representation

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService, restore_root: str):
        fields_dict = cls._load_fields(dict_pers_service, file_pers_service, obj_id, restore_root)

        return cls(data_wrapper=fields_dict[DATA_WRAPPER], dataloader=fields_dict[DATA_LOADER],
                   pre_processor=fields_dict[PRE_PROCESSOR], environment=fields_dict[ENVIRONMENT],
                   loss=fields_dict[LOSS], optimizer=fields_dict[OPTIMIZER], store_id=fields_dict[ID])

    @classmethod
    def _load_fields(cls, dict_pers_service, file_pers_service, obj_id, restore_root):
        restored_dict = dict_pers_service.recover_dict(obj_id, TRAIN_INFO)

        result = super()._load_fields(dict_pers_service, file_pers_service, obj_id, restore_root)

        loss_id = restored_dict[LOSS]
        result[LOSS] = Function.load(loss_id, file_pers_service, dict_pers_service, restore_root)

        optimizer_id = restored_dict[OPTIMIZER]
        optimizer = RestorableObjectWrapper.load(optimizer_id, file_pers_service, dict_pers_service, restore_root)
        optimizer.restore_instance()
        result[OPTIMIZER] = optimizer

        return result

    def size_in_bytes(self, file_pers_service: AbstractFilePersistenceService,
                      dict_pers_service: AbstractDictPersistenceService) -> int:
        result = 0

        # size of the dict
        result += dict_pers_service.dict_size(self.store_id, TRAIN_INFO)

        result = super()._fields_size(dict_pers_service, file_pers_service)

        result += self.loss.size_in_bytes(file_pers_service, dict_pers_service)
        result += self.optimizer.size_in_bytes(file_pers_service, dict_pers_service)

        return result

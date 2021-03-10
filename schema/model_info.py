from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from mmlib.save_info import TrainInfo
from schema.inference_info import InferenceInfo
from schema.recover_info import AbstractRecoverInfo
from schema.schema_obj import SchemaObj

ID = 'id'
STORE_TYPE = 'store_type'
RECOVER_INFO_ID = 'recover_info_id'
DERIVED_FROM = 'derived_from'
INFERENCE_INFO_ID = 'inference_info_id'
TRAIN_INFO_ID = 'train_info_id'

REPRESENT_TYPE = 'model_info'


class ModelInfo(SchemaObj):

    def __init__(self, store_type: str, recover_info: AbstractRecoverInfo, store_id: str = None,
                 derived_from_id: str = None, inference_info: str = None, train_info: str = None):
        self.store_id = store_id
        self.store_type = store_type
        self.recover_info = recover_info
        self.derived_from = derived_from_id
        self.inference_info = inference_info
        self.train_info = train_info

    def persist(self, file_pers_service: AbstractFilePersistenceService,
                dict_pers_service: AbstractDictPersistenceService) -> str:

        if not self.store_id:
            self.store_id = dict_pers_service.generate_id()

        recover_info_id = self.recover_info.persist(file_pers_service, dict_pers_service)

        # add mandatory fields
        dict_representation = {
            ID: self.store_id,
            STORE_TYPE: self.store_type,
            RECOVER_INFO_ID: recover_info_id,
        }

        # add optional fields if set
        if self.derived_from:
            dict_representation[DERIVED_FROM] = self.derived_from
        if self.inference_info:
            dict_representation[INFERENCE_INFO_ID] = self.derived_from
        if self.train_info:
            dict_representation[TRAIN_INFO_ID] = self.derived_from

        dict_pers_service.save_dict(dict_representation, REPRESENT_TYPE)

        return self.store_id

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService, restore_root: str):

        restored_dict = dict_pers_service.recover_dict(obj_id, REPRESENT_TYPE)

        # mandatory fields
        store_type = restored_dict[STORE_TYPE]

        recover_info_id = restored_dict[RECOVER_INFO_ID]
        recover_info = AbstractRecoverInfo.load(recover_info_id, file_pers_service, dict_pers_service, restore_root)

        # optional fields
        derived_from_id = restored_dict[DERIVED_FROM] if DERIVED_FROM in restored_dict else None
        inference_info = None
        train_info = None

        if INFERENCE_INFO_ID in restored_dict:
            inference_info_id = restored_dict[INFERENCE_INFO_ID]
            inference_info = InferenceInfo.load(inference_info_id, file_pers_service, dict_pers_service, restore_root)

        if TRAIN_INFO_ID in restored_dict:
            train_info_id = restored_dict[TRAIN_INFO_ID]
            train_info = TrainInfo.load(train_info_id, file_pers_service, dict_pers_service, restore_root)

        return cls(store_type=store_type, recover_info=recover_info, store_id=obj_id, derived_from_id=derived_from_id,
                   inference_info=inference_info, train_info=train_info)

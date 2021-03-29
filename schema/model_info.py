from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from schema.inference_info import InferenceInfo
from schema.recover_info import AbstractRecoverInfo, FullModelRecoverInfo, ProvenanceRecoverInfo
from schema.schema_obj import SchemaObj
from schema.store_type import ModelStoreType
from schema.train_info import TrainInfo

ID = 'id'
STORE_TYPE = 'store_type'
RECOVER_INFO_ID = 'recover_info_id'
DERIVED_FROM = 'derived_from'
INFERENCE_INFO_ID = 'inference_info_id'
TRAIN_INFO_ID = 'train_info_id'

MODEL_INFO = 'model_info'


class ModelInfo(SchemaObj):


    def __init__(self, store_type: ModelStoreType, recover_info: AbstractRecoverInfo, store_id: str = None,
                 derived_from_id: str = None, inference_info: InferenceInfo = None, train_info: TrainInfo = None):
        super().__init__(store_id)
        self.store_type = store_type
        self.recover_info = recover_info
        self.derived_from = derived_from_id
        self.inference_info = inference_info
        self.train_info = train_info

    def persist(self, file_pers_service: AbstractFilePersistenceService,
                dict_pers_service: AbstractDictPersistenceService) -> str:

        super().persist(file_pers_service, dict_pers_service)

        recover_info_id = self.recover_info.persist(file_pers_service, dict_pers_service)
        print('recover_info_id')
        print(recover_info_id)

        # add mandatory fields
        dict_representation = {
            ID: self.store_id,
            STORE_TYPE: self.store_type.value,
            RECOVER_INFO_ID: recover_info_id,
        }

        # add optional fields if set
        if self.derived_from:
            dict_representation[DERIVED_FROM] = self.derived_from
        if self.inference_info:
            inference_info_id = self.inference_info.persist(file_pers_service, dict_pers_service)
            dict_representation[INFERENCE_INFO_ID] = inference_info_id
        if self.train_info:
            train_info_id = self.train_info.persist(file_pers_service, dict_pers_service)
            dict_representation[TRAIN_INFO_ID] = train_info_id

        dict_pers_service.save_dict(dict_representation, MODEL_INFO)

        return self.store_id

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService, restore_root: str):

        restored_dict = dict_pers_service.recover_dict(obj_id, MODEL_INFO)

        # mandatory fields
        store_type = ModelStoreType(restored_dict[STORE_TYPE])

        recover_info_id = restored_dict[RECOVER_INFO_ID]

        recover_info = None
        if store_type == ModelStoreType.PICKLED_WEIGHTS:
            recover_info = FullModelRecoverInfo.load(recover_info_id, file_pers_service, dict_pers_service,
                                                     restore_root)
        elif store_type == ModelStoreType.PROVENANCE:
            recover_info = ProvenanceRecoverInfo.load(recover_info_id, file_pers_service, dict_pers_service,
                                                      restore_root)
        else:
            assert False, 'Not implemented yet'

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

    def size_in_bytes(self, file_pers_service: AbstractFilePersistenceService,
                      dict_pers_service: AbstractDictPersistenceService) -> int:
        result = 0

        # size of the dict
        result += dict_pers_service.dict_size(self.store_id, MODEL_INFO)

        # size of all referenced files/objects
        # for now we leave out the size of the base model, we might have to implement this later
        result += self.recover_info.size_in_bytes(file_pers_service, dict_pers_service)
        if self.inference_info:
            result += self.inference_info.size_in_bytes(file_pers_service, dict_pers_service)
        if self.train_info:
            result += self.train_info.size_in_bytes(file_pers_service, dict_pers_service)

        return result

    def _representation_type(self) -> str:
        return MODEL_INFO


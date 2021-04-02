from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from schema.inference_info import InferenceInfo
from schema.recover_info import AbstractRecoverInfo, FullModelRecoverInfo, ProvenanceRecoverInfo
from schema.schema_obj import SchemaObj
from schema.store_type import ModelStoreType
from schema.train_info import TrainInfo

STORE_TYPE = 'store_type'
RECOVER_INFO_ID = 'recover_info_id'
DERIVED_FROM = 'derived_from'
INFERENCE_INFO_ID = 'inference_info_id'
TRAIN_INFO_ID = 'train_info_id'

MODEL_INFO = 'model_info'


class ModelInfo(SchemaObj):

    def __init__(self, store_type: ModelStoreType = None, recover_info: AbstractRecoverInfo = None,
                 store_id: str = None, derived_from_id: str = None, inference_info: InferenceInfo = None,
                 train_info: TrainInfo = None):
        super().__init__(store_id)
        self.store_type = store_type
        self.recover_info = recover_info
        self.derived_from = derived_from_id
        self.inference_info = inference_info
        self.train_info = train_info

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):

        recover_info_id = self.recover_info.persist(file_pers_service, dict_pers_service)
        print('recover_info_id')
        print(recover_info_id)

        # add mandatory fields
        dict_representation[STORE_TYPE] = self.store_type.value
        dict_representation[RECOVER_INFO_ID] = recover_info_id

        # add optional fields if set
        if self.derived_from:
            dict_representation[DERIVED_FROM] = self.derived_from
        if self.inference_info:
            inference_info_id = self.inference_info.persist(file_pers_service, dict_pers_service)
            dict_representation[INFERENCE_INFO_ID] = inference_info_id
        if self.train_info:
            train_info_id = self.train_info.persist(file_pers_service, dict_pers_service)
            dict_representation[TRAIN_INFO_ID] = train_info_id

    @classmethod
    def load_placeholder(cls, obj_id: str):
        return cls(store_id=obj_id)

    @classmethod
    def load(cls, obj_id: str, file_pers_service: AbstractFilePersistenceService,
             dict_pers_service: AbstractDictPersistenceService, restore_root: str, load_recursive: bool = False):

        print('xxxxxxxxxxxxxxxxxLOAD MODEL INFO ')

        restored_dict = _recover_stored_dict(dict_pers_service, obj_id)

        # mandatory fields
        store_type = _recover_store_type(restored_dict)
        recover_info = _recover_recover_info(restored_dict, dict_pers_service, file_pers_service, restore_root,
                                             store_type, load_recursive)

        # optional fields
        derived_from_id = _recover_derived_from(restored_dict)
        inference_info = None
        train_info = None

        # Note not implemented yet
        # if INFERENCE_INFO_ID in restored_dict:
        #     inference_info_id = restored_dict[INFERENCE_INFO_ID]
        #     inference_info = InferenceInfo.load(inference_info_id, file_pers_service, dict_pers_service, restore_root)
        #
        # if TRAIN_INFO_ID in restored_dict:
        #     train_info_id = restored_dict[TRAIN_INFO_ID]
        #     train_info = TrainInfo.load(train_info_id, file_pers_service, dict_pers_service, restore_root)

        return cls(store_type=store_type, recover_info=recover_info, store_id=obj_id, derived_from_id=derived_from_id,
                   inference_info=inference_info, train_info=train_info)

    def load_all_fields(self, file_pers_service: AbstractFilePersistenceService,
                        dict_pers_service: AbstractDictPersistenceService, restore_root: str, load_ref_fields=True):

        restored_dict = _recover_stored_dict(dict_pers_service, self.store_id)

        # mandatory fields
        if not self.store_type:
            self.store_type = _recover_store_type(restored_dict)

        if not self.recover_info:
            self.recover_info = _recover_recover_info(restored_dict, dict_pers_service, file_pers_service, restore_root,
                                                      self.store_type, load_recursive=not load_ref_fields)

        # optional fields
        if not self.derived_from:
            self.derived_from = _recover_derived_from(restored_dict)

        # NOTE: other fields not implemented yet

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


def _recover_stored_dict(dict_pers_service, obj_id):
    return dict_pers_service.recover_dict(obj_id, MODEL_INFO)


def _recover_store_type(restored_dict):
    return ModelStoreType(restored_dict[STORE_TYPE])


def _recover_recover_info(restored_dict, dict_pers_service, file_pers_service, restore_root, store_type,
                          load_recursive):
    recover_info_id = restored_dict[RECOVER_INFO_ID]

    if store_type == ModelStoreType.PICKLED_WEIGHTS:
        if load_recursive:
            recover_info = FullModelRecoverInfo.load(recover_info_id, file_pers_service, dict_pers_service,
                                                     restore_root)
        else:
            recover_info = FullModelRecoverInfo.load_placeholder(recover_info_id)
    elif store_type == ModelStoreType.PROVENANCE:
        if load_recursive:
            recover_info = ProvenanceRecoverInfo.load(recover_info_id, file_pers_service, dict_pers_service,
                                                      restore_root)
        else:
            recover_info = ProvenanceRecoverInfo.load_placeholder(recover_info_id)
    else:
        assert False, 'Not implemented yet'
    return recover_info


def _recover_derived_from(restored_dict):
    return restored_dict[DERIVED_FROM] if DERIVED_FROM in restored_dict else None

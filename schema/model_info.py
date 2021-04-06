from mmlib.persistence import FilePersistenceService, DictPersistenceService
from schema.recover_info import AbstractRecoverInfo, FullModelRecoverInfo, ProvenanceRecoverInfo
from schema.schema_obj import SchemaObj
from schema.store_type import ModelStoreType

STORE_TYPE = 'store_type'
RECOVER_INFO_ID = 'recover_info_id'
DERIVED_FROM = 'derived_from'
INFERENCE_INFO_ID = 'inference_info_id'
TRAIN_INFO_ID = 'train_info_id'

MODEL_INFO = 'model_info'


class ModelInfo(SchemaObj):

    def __init__(self, store_type: ModelStoreType = None, recover_info: AbstractRecoverInfo = None,
                 store_id: str = None, derived_from_id: str = None):
        super().__init__(store_id)
        self.store_type = store_type
        self.recover_info = recover_info
        self.derived_from = derived_from_id

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

    def load_all_fields(self, file_pers_service: FilePersistenceService,
                        dict_pers_service: DictPersistenceService, restore_root: str,
                        load_recursive: bool = False, load_files: bool = False):

        restored_dict = _recover_stored_dict(dict_pers_service, self.store_id)

        # mandatory fields
        if not self.store_type:
            self.store_type = _recover_store_type(restored_dict)

        if not self.recover_info:
            self.recover_info = _recover_recover_info(restored_dict, dict_pers_service, file_pers_service, restore_root,
                                                      self.store_type, load_recursive, load_files)

        # optional fields
        if not self.derived_from:
            self.derived_from = _recover_derived_from(restored_dict)

    def size_in_bytes(self, file_pers_service: FilePersistenceService,
                      dict_pers_service: DictPersistenceService) -> int:
        result = 0

        # size of the dict
        result += dict_pers_service.dict_size(self.store_id, MODEL_INFO)

        # size of all referenced files/objects
        # for now we leave out the size of the base model, we might have to implement this later
        result += self.recover_info.size_in_bytes(file_pers_service, dict_pers_service)

        return result

    def _representation_type(self) -> str:
        return MODEL_INFO


def _recover_stored_dict(dict_pers_service, obj_id):
    return dict_pers_service.recover_dict(obj_id, MODEL_INFO)


def _recover_store_type(restored_dict):
    return ModelStoreType(restored_dict[STORE_TYPE])


def _recover_recover_info(restored_dict, dict_pers_service, file_pers_service, restore_root, store_type,
                          load_recursive, load_files):
    recover_info_id = restored_dict[RECOVER_INFO_ID]

    if store_type == ModelStoreType.PICKLED_WEIGHTS:
        if load_recursive:
            recover_info = FullModelRecoverInfo.load(recover_info_id, file_pers_service, dict_pers_service,
                                                     restore_root, load_recursive, load_files)
        else:
            recover_info = FullModelRecoverInfo.load_placeholder(recover_info_id)
    elif store_type == ModelStoreType.PROVENANCE:
        if load_recursive:
            recover_info = ProvenanceRecoverInfo.load(recover_info_id, file_pers_service, dict_pers_service,
                                                      restore_root, load_recursive, load_files)
        else:
            recover_info = ProvenanceRecoverInfo.load_placeholder(recover_info_id)
    else:
        assert False, 'Not implemented yet'
    return recover_info


def _recover_derived_from(restored_dict):
    return restored_dict[DERIVED_FROM] if DERIVED_FROM in restored_dict else None

import os

from mmlib.persistence import FilePersistenceService, DictPersistenceService
from schema.file_reference import FileReference
from schema.restorable_object import StateDictRestorableObjectWrapper
from schema.schema_obj import SchemaObj
from util.init_from_file import create_type

TRAIN_SERVICE = 'train_service'
TRAIN_KWARGS = 'train_kwargs'
WRAPPER_CODE = 'wrapper_code'
WRAPPER_CLASS_NAME = 'wrapper_class_name'

TRAIN_INFO = 'train_info'


class TrainInfo(SchemaObj):

    def _add_reference_sizes(self, size_dict, file_pers_service, dict_pers_service):
        # TODO
        pass

    def __init__(self, ts_wrapper: StateDictRestorableObjectWrapper = None, ts_wrapper_code: FileReference = None,
                 ts_wrapper_class_name: str = None, train_kwargs: dict = None, store_id: str = None):
        super().__init__(store_id)
        self.train_service_wrapper = ts_wrapper
        self.train_service_wrapper_code = ts_wrapper_code
        self.train_service_wrapper_class_name = ts_wrapper_class_name
        self.train_kwargs = train_kwargs

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        train_service_id = self.train_service_wrapper.persist(file_pers_service, dict_pers_service)
        file_pers_service.save_file(self.train_service_wrapper_code)

        dict_representation[TRAIN_SERVICE] = train_service_id
        dict_representation[WRAPPER_CODE] = self.train_service_wrapper_code.reference_id
        dict_representation[WRAPPER_CLASS_NAME] = self.train_service_wrapper_class_name
        dict_representation[TRAIN_KWARGS] = self.train_kwargs

    def load_all_fields(self, file_pers_service: FilePersistenceService,
                        dict_pers_service: DictPersistenceService, restore_root: str,
                        load_recursive: bool = True, load_files: bool = True):
        restored_dict = dict_pers_service.recover_dict(self.store_id, TRAIN_INFO)

        self.train_service_wrapper_class_name = restored_dict[WRAPPER_CLASS_NAME]
        self.train_kwargs = restored_dict[TRAIN_KWARGS]

        train_service_id = restored_dict[TRAIN_SERVICE]
        self.train_service_wrapper_code = FileReference(reference_id=restored_dict[WRAPPER_CODE])
        self.train_service_wrapper = \
            _recover_train_service_wrapper(dict_pers_service, file_pers_service, restore_root,
                                           train_service_id, self.train_service_wrapper_class_name,
                                           self.train_service_wrapper_code, load_recursive, load_files)

    def _size_class_specific_fields(self, file_pers_service, dict_pers_service):
        result = 0

        result += self.train_service_wrapper.size_in_bytes(file_pers_service, dict_pers_service)
        result += file_pers_service.size(self.train_service_wrapper_code)

        return result

    @property
    def _representation_type(self) -> str:
        return TRAIN_INFO


def _recover_train_service_wrapper(dict_pers_service, file_pers_service, restore_root, train_service_id,
                                   ts_wrapper_class_name, ts_wrapper_code, load_recursive,
                                   load_files):
    # TODO: Future work think about better solution without loading code
    # TODO maybe can be replaced when using FileRef Object
    restore_dir = os.path.join(restore_root, 'RESTORE_PATH')
    os.mkdir(restore_dir)
    file_pers_service.recover_file(ts_wrapper_code, restore_dir)
    wrapper_class = create_type(code=ts_wrapper_code.path, type_name=ts_wrapper_class_name)
    if load_recursive:
        train_service_wrapper = wrapper_class.load(train_service_id, file_pers_service,
                                                   dict_pers_service, restore_root, load_recursive, load_files)
        train_service_wrapper.restore_instance(file_pers_service, dict_pers_service, restore_root)
    else:
        train_service_wrapper = wrapper_class.load_placeholder(train_service_id)
    return train_service_wrapper

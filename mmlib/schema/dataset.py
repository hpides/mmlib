import os

from mmlib.persistence import FilePersistenceService, DictPersistenceService
from mmlib.schema.file_reference import FileReference
from mmlib.schema.schema_obj import SchemaObj
from util.zip import zip_path, unzip

RAW_DATA = 'raw_data'

DATASET = 'dataset'


class Dataset(SchemaObj):

    def __init__(self, raw_data: FileReference = None, store_id: str = None):
        super().__init__(store_id)
        self.raw_data = raw_data

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        raw_data_path = self.raw_data.path
        zip_file_path = zip_path(raw_data_path)
        zipped_raw_data = FileReference(path=zip_file_path)
        file_pers_service.save_file(zipped_raw_data)

        dict_representation[RAW_DATA] = zipped_raw_data.reference_id

    def load_all_fields(self, file_pers_service: FilePersistenceService,
                        dict_pers_service: DictPersistenceService, restore_root: str,
                        load_recursive: bool = True, load_files: bool = True):
        restored_dict = dict_pers_service.recover_dict(self.store_id, DATASET)
        self.raw_data = _recover_data(file_pers_service, load_files, restore_root, restored_dict)

    @property
    def _representation_type(self) -> str:
        return DATASET

    def _add_reference_sizes(self, size_dict, file_pers_service, dict_pers_service):
        file_pers_service.file_size(self.raw_data)
        size_dict[RAW_DATA] = self.raw_data.size


def _recover_data(file_pers_service, load_files, restore_root, restored_dict):
    raw_data_id = restored_dict[RAW_DATA]
    raw_data = FileReference(reference_id=raw_data_id)

    # TODO create zippable file ref
    if load_files:
        file_pers_service.recover_file(raw_data, restore_root)
        zip_file_path = raw_data.path
        restore_path = os.path.join(restore_root, 'data')
        unzipped_raw_data = unzip(zip_file_path, restore_path)
        raw_data = FileReference(path=unzipped_raw_data)
    return raw_data

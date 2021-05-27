import os
import shutil

from schema.dataset import Dataset
from schema.file_reference import FileReference
from tests.size.abstract_test_size import TestSize

COCO_DATA = '../example_files/data/reduced-custom-coco-data'

MONGO_CONTAINER_NAME = 'mongo-test'


class TestDatasetSize(TestSize):

    def __clean_up(self):
        os.system('docker kill %s' % MONGO_CONTAINER_NAME)
        if os.path.exists(self.abs_tmp_path):
            shutil.rmtree(self.abs_tmp_path)

        if os.path.exists(COCO_DATA + '.zip'):
            os.remove(COCO_DATA + '.zip')

    def test_dataset_size(self):
        file = FileReference(path=COCO_DATA)
        data_set = Dataset(raw_data=file)
        data_set_id = data_set.persist(self.file_pers_service, self.dict_pers_service)

        place_holder = Dataset.load_placeholder(data_set_id)
        size_dict = place_holder.size_info(self.file_pers_service, self.dict_pers_service)

        # raw data number from mac finder info
        self.assertEqual(len(size_dict.keys()), 2)
        self.assertTrue('metadata_size' in size_dict.keys())
        self.assertTrue('raw_data' in size_dict.keys())
        self.assertTrue(size_dict['metadata_size'] > 0)
        # file should be bigger than 22 MB
        self.assertTrue(size_dict['raw_data'] > 22000000)

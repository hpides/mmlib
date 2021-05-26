import os
import shutil
import unittest

from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from schema.dataset import Dataset
from schema.file_reference import FileReference
from util.mongo import MongoService

MONGO_CONTAINER_NAME = 'mongo-test'


class TestDatasetSize(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_path = './filesystem-tmp'
        self.abs_tmp_path = os.path.abspath(self.tmp_path)

        self.__clean_up()

        # run mongo DB locally in docker container
        os.system('docker run --rm --name %s -it -p 27017:27017 -d  mongo:4.4.3 ' % MONGO_CONTAINER_NAME)

        self.mongo_service = MongoService('127.0.0.1', 'mmlib')

        os.mkdir(self.abs_tmp_path)
        self.file_pers_service = FileSystemPersistenceService(self.tmp_path)
        self.dict_pers_service = MongoDictPersistenceService()


    def tearDown(self) -> None:
        self.__clean_up()

    def __clean_up(self):
        os.system('docker kill %s' % MONGO_CONTAINER_NAME)
        if os.path.exists(self.abs_tmp_path):
            shutil.rmtree(self.abs_tmp_path)

    def test_dataset_size(self):
        file = FileReference(path='../example_files/data/reduced-custom-coco-data')
        data_set = Dataset(raw_data=file)
        data_set_id = data_set.persist(self.file_pers_service, self.dict_pers_service)

        place_holder = Dataset.load_placeholder(data_set_id)
        size_dict = place_holder.size_info(self.file_pers_service, self.dict_pers_service)

        # raw data number form mac finder info
        expected = {'metadata_size': 99, 'raw_data': 22336767}
        self.assertEqual(size_dict, expected)

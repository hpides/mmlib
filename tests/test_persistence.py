import filecmp
import os
import shutil
import unittest

from mmlib.persistence import FileSystemMongoPS

MONGO_CONTAINER_NAME = 'mongo-test'
TEST_COLLECTION = 'test-collection'


class TestPersistence(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_path = './filesystem-tmp'
        self.abs_tmp_path = os.path.abspath(self.tmp_path)
        self.tmp_dir = './tmp'
        self.abs_save_service_tmp = os.path.abspath(self.tmp_dir)

        self.__clean_up()
        # run mongo DB locally in docker container
        os.system('docker run --rm --name %s -it -p 27017:27017 -d  mongo:4.4.3 ' % MONGO_CONTAINER_NAME)

        os.mkdir(self.abs_tmp_path)
        os.mkdir(self.abs_save_service_tmp)
        self.pers_service = FileSystemMongoPS(self.tmp_path)

    def tearDown(self) -> None:
        self.__clean_up()

    def __clean_up(self):
        os.system('docker kill %s' % MONGO_CONTAINER_NAME)
        if os.path.exists(self.abs_tmp_path):
            shutil.rmtree(self.abs_tmp_path)
        if os.path.exists(self.abs_save_service_tmp):
            shutil.rmtree(self.abs_save_service_tmp)

    def test_save_recover_dict(self):
        test_dict = {'test': 'test'}
        dict_id = self.pers_service.save_dict(test_dict, TEST_COLLECTION)

        recovered_dict = self.pers_service.recover_dict(dict_id, TEST_COLLECTION)

        self.assertEqual(test_dict, recovered_dict)

    def test_save_recover_file(self):
        file_name = 'test-file.txt'
        file_path = './test-files/test-file.txt'
        file_id = self.pers_service.save_file(file_path)
        self.pers_service.recover_file(file_id, self.tmp_dir)

        self.assertTrue(filecmp.cmp(file_path, os.path.join(self.tmp_dir, file_name)))

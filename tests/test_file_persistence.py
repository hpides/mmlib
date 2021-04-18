import filecmp
import os
import shutil
import unittest

from mmlib.persistence import FileSystemPersistenceService
from schema.file_reference import FileReference


class TestPersistence(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_path = './filesystem-tmp'
        self.abs_tmp_path = os.path.abspath(self.tmp_path)
        self.tmp_dir = './tmp'
        self.abs_save_service_tmp = os.path.abspath(self.tmp_dir)

        self.__clean_up()

        os.mkdir(self.abs_tmp_path)
        os.mkdir(self.abs_save_service_tmp)
        self.pers_service = FileSystemPersistenceService(self.tmp_path)

    def tearDown(self) -> None:
        self.__clean_up()

    def __clean_up(self):
        if os.path.exists(self.abs_tmp_path):
            shutil.rmtree(self.abs_tmp_path)
        if os.path.exists(self.abs_save_service_tmp):
            shutil.rmtree(self.abs_save_service_tmp)

    def test_save_recover_file(self):
        file_path = './test-files/test-file.txt'
        file_ref = FileReference(path=file_path)
        self.pers_service.save_file(file_ref)
        self.pers_service.recover_file(file_ref, self.tmp_dir)

        self.assertTrue(filecmp.cmp(file_path, file_ref.path))

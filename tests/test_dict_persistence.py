import os
import unittest

from mmlib.persistence import MongoDictPersistenceService

MONGO_CONTAINER_NAME = 'mongo-test'
TEST_COLLECTION = 'test-collection'


class TestPersistence(unittest.TestCase):

    def setUp(self) -> None:
        self.__clean_up()
        # run mongo DB locally in docker container
        os.system('docker run --rm --name %s -it -p 27017:27017 -d  mongo:4.4.3 ' % MONGO_CONTAINER_NAME)

        self.pers_service = MongoDictPersistenceService()

    def tearDown(self) -> None:
        self.__clean_up()

    def __clean_up(self):
        os.system('docker kill %s' % MONGO_CONTAINER_NAME)

    def test_save_recover_dict(self):
        test_dict = {'test': 'test'}
        dict_id = self.pers_service.save_dict(test_dict, TEST_COLLECTION)

        recovered_dict = self.pers_service.recover_dict(dict_id, TEST_COLLECTION)

        self.assertEqual(test_dict, recovered_dict)

import os
import unittest

from mmlib.save import MongoService

MONGO_CONTAINER_NAME = 'mongo-test'


class TestProbe(unittest.TestCase):

    def setUp(self) -> None:
        os.system('docker run --rm --name %s -it -p 27017:27017 -d  mongo:latest' % MONGO_CONTAINER_NAME)
        self.mongo_service = MongoService('127.0.0.1')

    def tearDown(self) -> None:
        os.system('docker kill %s' % MONGO_CONTAINER_NAME)

    def test_save_json(self):
        test_dict = {'test': 'test'}
        model_id = self.mongo_service.save_dict(test_dict)
        expected_dict = test_dict
        expected_dict['_id'] = model_id

        num_entries = len(self.mongo_service.get_ids())
        retrieve = self.mongo_service.get_model_dict(model_id=model_id)

        self.assertEqual(1, num_entries)
        self.assertEqual(expected_dict, retrieve)

    def test_add_attribute(self):
        test_dict = {'test': 'test'}
        model_id = self.mongo_service.save_dict(test_dict)

        add = {'added': 'added'}
        self.mongo_service.add_attribute(model_id, add)

        expected_dict = test_dict
        expected_dict['_id'] = model_id
        expected_dict.update(add)

        retrieve = self.mongo_service.get_model_dict(model_id=model_id)

        self.assertEqual(expected_dict, retrieve)

import os
import shutil
import unittest

from torchvision import models

from mmlib.save import MongoService, SaveService, SaveType

MONGO_CONTAINER_NAME = 'mongo-test'


class TestProbe(unittest.TestCase):

    def setUp(self) -> None:
        # run mongo DB locally in docker container
        os.system('docker run --rm --name %s -it -p 27017:27017 -d  mongo:latest' % MONGO_CONTAINER_NAME)

        self.mongo_service = MongoService('127.0.0.1')

        self.abs_tmp_path = os.path.abspath('./tmp')

        os.mkdir(self.abs_tmp_path)
        self.save_service = SaveService(self.abs_tmp_path)

    def tearDown(self) -> None:
        os.system('docker kill %s' % MONGO_CONTAINER_NAME)

        shutil.rmtree(self.abs_tmp_path)

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

    def test_save_model(self):
        model = models.resnet18(pretrained=True)

        model_id = self.save_service.save_model('test_model', model)

        expected_dict = {
            '_id': model_id,
            'name': 'test_model',
            'store-type': SaveType.PICKLED_MODEL.value,
            'save-path': os.path.join(self.save_service._base_path, str(model_id))
        }

        retrieve = self.mongo_service.get_model_dict(model_id=model_id)
        self.assertEqual(expected_dict, retrieve)

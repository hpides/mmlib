import os
import shutil
import unittest

from bson import ObjectId
from torchvision import models

from mmlib.equal import model_equal
from mmlib.helper import imagenet_input
from mmlib.save import FileSystemMongoSaveRecoverService, SaveType
from tests.networks.mynets.test_net import TestNet
from util.mongo import MongoService

MONGO_CONTAINER_NAME = 'mongo-test'


class TestSave(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_path = './tmp'
        self.abs_tmp_path = os.path.abspath(self.tmp_path)

        self.__clean_up()
        # run mongo DB locally in docker container
        os.system('docker run --rm --name %s -it -p 27017:27017 -d  mongo:4.4.3 ' % MONGO_CONTAINER_NAME)

        self.mongo_service = MongoService('127.0.0.1', 'mmlib', 'models')

        os.mkdir(self.abs_tmp_path)
        self.save_recover_service = FileSystemMongoSaveRecoverService(self.abs_tmp_path)

    def tearDown(self) -> None:
        self.__clean_up()

    def __clean_up(self):
        os.system('docker kill %s' % MONGO_CONTAINER_NAME)
        if os.path.exists(self.abs_tmp_path):
            shutil.rmtree(self.abs_tmp_path)

    def test_save_json(self):
        test_dict = {'test': 'test'}
        model_id = self.mongo_service.save_dict(test_dict)
        expected_dict = test_dict
        expected_dict['_id'] = model_id

        num_entries = len(self.mongo_service.get_ids())
        retrieve = self.mongo_service.get_dict(object_id=model_id)

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

        retrieve = self.mongo_service.get_dict(object_id=model_id)

        self.assertEqual(expected_dict, retrieve)

    def test_save_model(self):
        model = models.resnet18(pretrained=True)

        model_id = self.save_recover_service.save_model('test_model', model, './networks/mynets/test_net.py', './..')
        obj_id = ObjectId(model_id)

        expected_dict = {
            '_id': obj_id,
            'name': 'test_model',
            'save-type': SaveType.PICKLED_MODEL.value,
            'save-path': os.path.join(self.save_recover_service._base_path, str(model_id) + '.zip')
        }

        retrieve = self.mongo_service.get_dict(object_id=obj_id)
        self.assertEqual(expected_dict, retrieve)

    def test_get_saved_ids(self):
        expected = []

        ids = self.save_recover_service.saved_model_ids()
        self.assertEqual(ids, expected)

        model = models.resnet18(pretrained=True)
        model_id = self.save_recover_service.save_model('test_model', model, './networks/mynets/test_net.py', './..')
        expected.append(model_id)

        ids = self.save_recover_service.saved_model_ids()
        self.assertEqual(ids, expected)

        model = models.resnet18(pretrained=True)
        model_id = self.save_recover_service.save_model('test_model', model, './networks/mynets/test_net.py', './..')
        expected.append(model_id)

        ids = self.save_recover_service.saved_model_ids()
        self.assertEqual(ids, expected)

    def test_save_and_restore(self):
        model = TestNet()
        model_id = self.save_recover_service.save_model('test_model', model, './networks/mynets/test_net.py', './..')

        # TODO test restore also on other machine
        restored_model = self.save_recover_service.recover_model(model_id)

        self.assertTrue(model_equal(model, restored_model, imagenet_input))

    def test_model_save_size(self):
        model = TestNet()
        model_id = self.save_recover_service.save_model('test_model', model, './networks/mynets/test_net.py', './..')

        save_size = self.save_recover_service.model_save_size(model_id)

        # got number form mac os finder file size info
        zip_size = 52242909

        met_data_size = self.mongo_service.document_size(ObjectId(model_id))

        self.assertEqual(met_data_size + zip_size, save_size)

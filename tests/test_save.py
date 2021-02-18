import os
import shutil
import unittest

from bson import ObjectId

from mmlib.deterministic import set_deterministic
from mmlib.equal import model_equal
from mmlib.helper import imagenet_input
from mmlib.persistence import FileSystemMongoPS
from mmlib.save import SimpleSaveRecoverService
from tests.networks.mynets.resnet18 import resnet18
from tests.networks.mynets.test_net import TestNet
from util.mongo import MongoService

MONGO_CONTAINER_NAME = 'mongo-test'


class TestSave(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_path = './filesystem-tmp'
        self.abs_tmp_path = os.path.abspath(self.tmp_path)
        self.save_service_tmp = './saveservice-tmp'
        self.abs_save_service_tmp = os.path.abspath(self.save_service_tmp)

        self.__clean_up()
        # run mongo DB locally in docker container
        os.system('docker run --rm --name %s -it -p 27017:27017 -d  mongo:4.4.3 ' % MONGO_CONTAINER_NAME)

        self.mongo_service = MongoService('127.0.0.1', 'mmlib')

        os.mkdir(self.abs_tmp_path)
        pers_service = FileSystemMongoPS(self.tmp_path)
        self.save_recover_service = SimpleSaveRecoverService(pers_service, self.save_service_tmp)

    def tearDown(self) -> None:
        self.__clean_up()

    def __clean_up(self):
        os.system('docker kill %s' % MONGO_CONTAINER_NAME)
        if os.path.exists(self.abs_tmp_path):
            shutil.rmtree(self.abs_tmp_path)
        if os.path.exists(self.abs_save_service_tmp):
            shutil.rmtree(self.abs_save_service_tmp)

    # def test_save_restore_model(self):
    #     model = googlenet()
    #
    #     model_id = self.save_recover_service.save_model('test_model', 'googlenet', model,
    #                                                     './networks/mynets/test_net.py')
    #
    #     restored_model = self.save_recover_service.recover_model(model_id)
    #
    #     self.assertTrue(model_equal(model, restored_model, imagenet_input))

    def test_save_restore_model(self):
        model = resnet18(pretrained=True)

        model_id = self.save_recover_service.save_model('test_model', 'resnet18', model,
                                                        './networks/mynets/resnet18.py')

        restored_model = self.save_recover_service.recover_model(model_id)

        self.assertTrue(model_equal(model, restored_model, imagenet_input))

    def test_save_restore_model_version(self):
        set_deterministic()
        model = resnet18()

        model_id = self.save_recover_service.save_model('test_model', 'resnet18', model,
                                                        './networks/mynets/resnet18.py')

        set_deterministic()
        model_version = resnet18()
        model_version1_id = self.save_recover_service.save_version(model_version, base_model_id=model_id)

        model_version = resnet18(pretrained=True)
        model_version2_id = self.save_recover_service.save_version(model_version, base_model_id=model_version1_id)

        restored_model = self.save_recover_service.recover_model(model_id)
        restored_model_version1 = self.save_recover_service.recover_model(model_version1_id)
        restored_model_version2 = self.save_recover_service.recover_model(model_version2_id)

        self.assertTrue(model_equal(model, restored_model, imagenet_input))
        self.assertTrue(model_equal(model, restored_model_version1, imagenet_input))
        self.assertFalse(model_equal(restored_model_version1, restored_model_version2, imagenet_input))

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

    def test_model_save_size(self):
        model = TestNet()
        model_id = self.save_recover_service.save_model('test_model', model, './networks/mynets/test_net.py', './..')

        save_size = self.save_recover_service.model_save_size(model_id)

        # got number form mac os finder file size info
        zip_size = 52242909

        met_data_size = self.mongo_service.document_size(ObjectId(model_id))

        self.assertEqual(met_data_size + zip_size, save_size)

import os
import shutil
import unittest

from bson import ObjectId

from mmlib.deterministic import set_deterministic
from mmlib.equal import model_equal
from mmlib.persistence import FileSystemMongoPS, DICT
from mmlib.save import SimpleSaveRecoverService
from schema.model_info import RECOVER_INFO
from schema.recover_info_t1 import RECOVER_VAL
from schema.schema_obj import SchemaObjType
from tests.networks.mynets.googlenet import googlenet
from tests.networks.mynets.mobilenet import mobilenet_v2
from tests.networks.mynets.resnet18 import resnet18
from util.dummy_data import imagenet_input
from util.mongo import MongoService

MONGO_CONTAINER_NAME = 'mongo-test'


class TestSave(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_path = './filesystem-tmp'
        self.abs_tmp_path = os.path.abspath(self.tmp_path)

        self.__clean_up()
        # run mongo DB locally in docker container
        os.system('docker run --rm --name %s -it -p 27017:27017 -d  mongo:4.4.3 ' % MONGO_CONTAINER_NAME)

        self.mongo_service = MongoService('127.0.0.1', 'mmlib')

        os.mkdir(self.abs_tmp_path)
        pers_service = FileSystemMongoPS(self.tmp_path)
        self.save_recover_service = SimpleSaveRecoverService(pers_service)

    def tearDown(self) -> None:
        self.__clean_up()

    def __clean_up(self):
        os.system('docker kill %s' % MONGO_CONTAINER_NAME)
        if os.path.exists(self.abs_tmp_path):
            shutil.rmtree(self.abs_tmp_path)

    def test_save_restore_model_googlenet(self):
        model = googlenet(aux_logits=True)

        self._test_save_restore_model('./networks/mynets/googlenet.py', 'googlenet', model)

    def test_save_restore_model_pretrained(self):
        file_names = ['mobilenet', 'resnet18']
        models = [mobilenet_v2, resnet18]
        for file_name, model in zip(file_names, models):
            code_name = model.__name__
            model = model(pretrained=True)
            code_file = './networks/mynets/{}.py'.format(file_name)

            self._test_save_restore_model(code_file, code_name, model)

    def test_save_restore_model(self):
        file_names = ['mobilenet', 'resnet18']
        models = [mobilenet_v2, resnet18]
        for file_name, model in zip(file_names, models):
            code_name = model.__name__
            model = model()
            code_file = './networks/mynets/{}.py'.format(file_name)

            self._test_save_restore_model(code_file, code_name, model)

    def _test_save_restore_model(self, code_file, code_name, model):
        model_id = self.save_recover_service.save_model(model, code_file, code_name)
        restored_model = self.save_recover_service.recover_model(model_id)
        self.assertTrue(model_equal(model, restored_model, imagenet_input))

    def test_save_restore_model_version(self):
        set_deterministic()
        model = resnet18()

        model_id = self.save_recover_service.save_model(model, './networks/mynets/resnet18.py', 'resnet18')

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

    def test_save_restore_model_and_recover_val(self):
        set_deterministic()
        model = resnet18()

        model_id = self.save_recover_service.save_model(
            model, './networks/mynets/resnet18.py', 'resnet18', recover_val=True, dummy_input_shape=[10, 3, 300, 400]
        )

        restored_model = self.save_recover_service.recover_model(model_id, check_recover_val=True)

        self.assertTrue(model_equal(model, restored_model, imagenet_input))

    def test_save_restore_model_version_and_recover_val(self):
        set_deterministic()
        model = resnet18()

        model_id = self.save_recover_service.save_model(
            model, './networks/mynets/resnet18.py', 'resnet18', recover_val=False
        )

        set_deterministic()
        model_version = resnet18()
        model_version1_id = self.save_recover_service.save_version(
            model_version, base_model_id=model_id, recover_val=True, dummy_input_shape=[10, 3, 300, 400]
        )

        model_version = resnet18(pretrained=True)
        # dummy_input_shape should be inferred form base model
        model_version2_id = self.save_recover_service.save_version(
            model_version, base_model_id=model_version1_id, recover_val=True
        )

        restored_model = self.save_recover_service.recover_model(model_id)
        restored_model_version1 = self.save_recover_service.recover_model(model_version1_id, check_recover_val=True)
        restored_model_version2 = self.save_recover_service.recover_model(model_version2_id, check_recover_val=True)

        self.assertTrue(model_equal(model, restored_model, imagenet_input))
        self.assertTrue(model_equal(model, restored_model_version1, imagenet_input))
        self.assertFalse(model_equal(restored_model_version1, restored_model_version2, imagenet_input))

    def test_save_restore_model_version_and_recover_val_assert_false(self):
        set_deterministic()
        model = resnet18()

        model_id = self.save_recover_service.save_model(
            model, './networks/mynets/resnet18.py', 'resnet18', recover_val=False
        )

        set_deterministic()
        model_version = resnet18()
        # we expect an exception because the dummy_input_shape is not given and can also not be inferred
        with self.assertRaises(Exception) as context:
            model_version1_id = self.save_recover_service.save_version(
                model_version, base_model_id=model_id, recover_val=True
            )

    def test_get_saved_ids(self):
        expected = []
        model = resnet18()

        ids = self.save_recover_service.saved_model_ids()
        self.assertEqual(ids, expected)

        model_id = self.save_recover_service.save_model(model, './networks/mynets/resnet18.py', 'resnet18')
        expected.append(model_id)

        ids = self.save_recover_service.saved_model_ids()
        self.assertEqual(ids, expected)

        model_id = self.save_recover_service.save_model(model, './networks/mynets/resnet18.py', 'resnet18')
        expected.append(model_id)

        ids = self.save_recover_service.saved_model_ids()
        self.assertEqual(ids, expected)

    def test_get_model_infos(self):
        expected = set()
        model = resnet18()

        model_infos = self.save_recover_service.saved_model_infos()
        self.assertEqual(set(model_infos), expected)

        model_id = self.save_recover_service.save_model(model, './networks/mynets/resnet18.py', 'resnet18')
        expected.add(self.save_recover_service._get_model_info(model_id))

        model_infos = self.save_recover_service.saved_model_infos()
        self.assertEqual(set(model_infos), expected)

        model_id = self.save_recover_service.save_model(model, './networks/mynets/resnet18.py', 'resnet18')
        expected.add(self.save_recover_service._get_model_info(model_id))

        model_infos = self.save_recover_service.saved_model_infos()
        self.assertEqual(set(model_infos), expected)

    def test_model_save_size(self):
        self._test_model_save_size()

    def test_model_save_size_recover_val(self):
        self._test_model_save_size(recover_val=True)

    def _test_model_save_size(self, recover_val=False):
        model = resnet18(pretrained=True)
        if recover_val:
            model_id = self.save_recover_service.save_model(
                model, './networks/mynets/resnet18.py', 'resnet18', recover_val=True,
                dummy_input_shape=[10, 3, 300, 400])
        else:
            model_id = self.save_recover_service.save_model(model, './networks/mynets/resnet18.py', 'resnet18')

        model_id = model_id.replace(DICT, '')
        save_size = self.save_recover_service.model_save_size(model_id)

        # got from os (macOS finder info)
        code_file_size = 6802
        pickled_weights_size = 46838023

        model_info_size = self.mongo_service.document_size(ObjectId(model_id), SchemaObjType.MODEL_INFO.value)
        model_info_dict = self.mongo_service.get_dict(ObjectId(model_id), SchemaObjType.MODEL_INFO.value)
        restore_info_id = model_info_dict[RECOVER_INFO].replace(DICT, '')
        restore_dict_size = self.mongo_service.document_size(ObjectId(restore_info_id), SchemaObjType.RECOVER_T1.value)

        # for now the size consists of
        #   - dict for model modelInfo
        #   - dict for restore info
        #   - dict for recoverVal info (ONLY IF SET)
        #   - pickled weights
        #   - code file
        expected_size = \
            code_file_size + \
            pickled_weights_size + \
            model_info_size + \
            restore_dict_size

        if recover_val:
            restore_info = self.mongo_service.get_dict(ObjectId(restore_info_id), SchemaObjType.RECOVER_T1.value)
            recover_val_id = restore_info[RECOVER_VAL].replace(DICT, '')
            val_dict_size = self.mongo_service.document_size(ObjectId(recover_val_id), SchemaObjType.RECOVER_VAL.value)
            expected_size += val_dict_size

        self.assertEqual(expected_size, save_size)

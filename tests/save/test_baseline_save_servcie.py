import os
import shutil
import unittest

from mmlib.equal import model_equal
from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from mmlib.recover_validation import RecoverValidationService
from mmlib.save import BaselineSaveService
from schema.save_info_builder import ModelSaveInfoBuilder
from tests.example_files.mynets.googlenet import googlenet
from tests.example_files.mynets.mobilenet import mobilenet_v2
from tests.example_files.mynets.resnet18 import resnet18
from util.dummy_data import imagenet_input
from util.mongo import MongoService

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
NETWORK_CODE_TEMPLATE = os.path.join(FILE_PATH, '../example_files/mynets/{}.py')
MONGO_CONTAINER_NAME = 'mongo-test'

DUMMY_INPUT_SHAPE = [10, 3, 300, 400]

GOOGLENET = 'googlenet'
MOBILENET = 'mobilenet'
RESNET_18 = 'resnet18'


class TestBaselineSaveService(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_path = './filesystem-tmp'
        self.abs_tmp_path = os.path.abspath(self.tmp_path)

        self.__clean_up()

        # run mongo DB locally in docker container
        os.system('docker run --rm --name %s -it -p 27017:27017 -d  mongo:4.4.3 ' % MONGO_CONTAINER_NAME)

        self.mongo_service = MongoService('127.0.0.1', 'mmlib')

        os.mkdir(self.abs_tmp_path)
        file_pers_service = FileSystemPersistenceService(self.tmp_path)
        dict_pers_service = MongoDictPersistenceService()
        self.recover_val_service = RecoverValidationService(dict_pers_service)
        self.init_save_service(dict_pers_service, file_pers_service)

    def init_save_service(self, dict_pers_service, file_pers_service):
        self.save_service = BaselineSaveService(file_pers_service, dict_pers_service)

    def tearDown(self) -> None:
        self.__clean_up()

    def __clean_up(self):
        os.system('docker kill %s' % MONGO_CONTAINER_NAME)
        if os.path.exists(self.abs_tmp_path):
            shutil.rmtree(self.abs_tmp_path)

    def test_save_restore_mobilenet(self):
        model = mobilenet_v2(pretrained=True)
        self._test_save_restore_model(model)

    def test_save_restore_resnet18(self):
        model = resnet18(pretrained=True)
        self._test_save_restore_model(model)

    def test_save_restore_model_googlenet(self):
        model = googlenet(aux_logits=True)
        self._test_save_restore_model(model)

    def _test_save_restore_model(self, model):
        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(model=model)
        save_info = save_info_builder.build()

        model_id = self.save_service.save_model(save_info)
        restored_model_info = self.save_service.recover_model(model_id)
        self.assertTrue(model_equal(model, restored_model_info.model, imagenet_input))

    def test_save_restore_mobilenet_val_info(self):
        model = mobilenet_v2(pretrained=True)
        self._test_save_restore_model_and_validation_info(model, DUMMY_INPUT_SHAPE)

    def test_save_restore_resnet18_val_info(self):
        model = resnet18(pretrained=True)
        self._test_save_restore_model_and_validation_info(model, DUMMY_INPUT_SHAPE)

    def test_save_restore_model_googlenet_val_info(self):
        model = googlenet(aux_logits=True)
        self._test_save_restore_model_and_validation_info(model, DUMMY_INPUT_SHAPE)

    def _test_save_restore_model_and_validation_info(self, model, dummy_input_shape):
        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(model=model)
        save_info = save_info_builder.build()

        model_id = self.save_service.save_model(save_info)

        # save additionally validation info
        self.save_service.save_validation_info(model, model_id, dummy_input_shape, self.recover_val_service)
        restored_model_info = self.save_service.recover_model(model_id, execute_checks=True,
                                                              recover_val_service=self.recover_val_service)
        self.assertTrue(model_equal(model, restored_model_info.model, imagenet_input))

    def test_save_restore_derived_models(self):
        initial_model = resnet18()

        # save initial model
        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(model=initial_model)
        save_info = save_info_builder.build()
        initial_model_id = self.save_service.save_model(save_info)

        # save derived model
        derived_model = resnet18(pretrained=True)
        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(model=derived_model, base_model_id=initial_model_id)
        save_info = save_info_builder.build()
        derived_model_id = self.save_service.save_model(save_info)

        restored_model_info = self.save_service.recover_model(derived_model_id)

        self.assertTrue(model_equal(derived_model, restored_model_info.model, imagenet_input))

    def test_save_restore_multiple_derived_models(self):
        initial_model = resnet18()

        # save initial model
        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(model=initial_model)
        save_info = save_info_builder.build()
        initial_model_id = self.save_service.save_model(save_info)

        # save derived model
        derived_model = resnet18(pretrained=True)
        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(model=derived_model, base_model_id=initial_model_id)
        save_info = save_info_builder.build()
        derived_model_id = self.save_service.save_model(save_info)

        restored_model_info = self.save_service.recover_model(derived_model_id)

        self.assertTrue(model_equal(derived_model, restored_model_info.model, imagenet_input))

        # save derived model
        derived_model_2 = restored_model_info.model
        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(model=derived_model_2, base_model_id=derived_model_id)
        save_info = save_info_builder.build()
        derived_model_id_2 = self.save_service.save_model(save_info)

        restored_model_info_2 = self.save_service.recover_model(derived_model_id_2)

        self.assertTrue(model_equal(derived_model_2, restored_model_info_2.model, imagenet_input))

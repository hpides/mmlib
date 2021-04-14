import os
import shutil
import unittest

from tests.networks.mynets.mobilenet import mobilenet_v2
from tests.networks.mynets.resnet18 import resnet18
from tests.test_baseline_save_servcie import MONGO_CONTAINER_NAME
from util.mongo import MongoService


class TestProvSaveService(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_path = './filesystem-tmp'
        self.abs_tmp_path = os.path.abspath(self.tmp_path)

        self.__clean_up()
        # run mongo DB locally in docker container
        os.system('docker run --rm --name %s -it -p 27017:27017 -d  mongo:4.4.3 ' % MONGO_CONTAINER_NAME)

        self.mongo_service = MongoService('127.0.0.1', 'mmlib')

        os.mkdir(self.abs_tmp_path)

    def tearDown(self) -> None:
        self.__clean_up()

    def __clean_up(self):
        os.system('docker kill %s' % MONGO_CONTAINER_NAME)
        if os.path.exists(self.abs_tmp_path):
            shutil.rmtree(self.abs_tmp_path)

    def test_save_restore_provenance_model_resnet18(self):
        model_name = resnet18.__name__
        self._test_save_restore_provenance_specific_model(model_name)

    def test_save_restore_provenance_model_mobilenet(self):
        model_name = mobilenet_v2.__name__
        self._test_save_restore_provenance_specific_model(model_name, filename='mobilenet')

    # googlenet has some problems when restored form state_dict with aux loss
    # NOTE think about not using googlenet for experiments
    # def test_save_restore_provenance_model_googlenet(self):
    #     model_name = googlenet.__name__
    #     self._test_save_restore_provenance_specific_model(model_name)

    def _test_save_restore_provenance_specific_model(self, model_name, filename=None):
        pass

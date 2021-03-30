import os
import unittest

from mmlib.persistence import MongoDictPersistenceService
from mmlib.recover_validation import RecoverValidationService
from tests.networks.mynets.resnet18 import resnet18
from tests.test_dict_persistence import MONGO_CONTAINER_NAME


class TestRecoverValidationService(unittest.TestCase):

    def setUp(self) -> None:
        os.system('docker run --rm --name %s -it -p 27017:27017 -d  mongo:4.4.3 ' % MONGO_CONTAINER_NAME)
        self.dict_pers_service = MongoDictPersistenceService()
        self.recover_val_service = RecoverValidationService(self.dict_pers_service)

    def tearDown(self) -> None:
        os.system('docker kill %s' % MONGO_CONTAINER_NAME)

    def test_check_recover_val_same_models(self):
        model1 = resnet18(pretrained=True)
        model_id = self.dict_pers_service.generate_id()
        self.recover_val_service.save_recover_val_info(model1, model_id, dummy_input_shape=[10, 3, 300, 400])

        self.assertTrue(self.recover_val_service.check_recover_val(model_id, model1))

    def test_check_recover_val_diff_models(self):
        model1 = resnet18()
        model_id = self.dict_pers_service.generate_id()
        self.recover_val_service.save_recover_val_info(model1, model_id, dummy_input_shape=[10, 3, 300, 400])

        model2 = resnet18()
        self.assertFalse(self.recover_val_service.check_recover_val(model_id, model2))

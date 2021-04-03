import os
import shutil
import unittest

import torch
from bson import ObjectId

from mmlib.constants import MMLIB_CONFIG, CURRENT_DATA_ROOT
from mmlib.deterministic import set_deterministic
from mmlib.equal import model_equal
from mmlib.persistence import MongoDictPersistenceService, FileSystemPersistenceService, DICT
from mmlib.recover_validation import RecoverValidationService
from mmlib.save import BaselineSaveService, ProvenanceSaveService
from schema.environment import Environment
from schema.model_info import RECOVER_INFO_ID, MODEL_INFO
from schema.recover_info import RECOVER_INFO
from schema.restorable_object import RestorableObjectWrapper, OptimizerWrapper
from schema.save_info_builder import ModelSaveInfoBuilder
from tests.inference_and_training.imagenet_train import ImagenetTrainService
from tests.networks.custom_coco import TrainCustomCoco
from tests.networks.mynets.googlenet import googlenet
from tests.networks.mynets.mobilenet import mobilenet_v2
from tests.networks.mynets.resnet152 import resnet152
from tests.networks.mynets.resnet18 import resnet18
from tests.networks.mynets.resnet50 import resnet50
from util.dummy_data import imagenet_input
from util.mongo import MongoService

MODEL_PATH = './networks/mynets/{}.py'

MONGO_CONTAINER_NAME = 'mongo-test'
COCO_ROOT = 'coco_root'
COCO_ANNOT = 'coco_annotations'

CONFIG = './local-config.ini'


class TestSave(unittest.TestCase):

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
        self.save_recover_service = BaselineSaveService(file_pers_service, dict_pers_service)
        self.provenance_save_service = ProvenanceSaveService(file_pers_service, dict_pers_service)
        self.recover_val_service = RecoverValidationService(dict_pers_service)

        os.environ[MMLIB_CONFIG] = CONFIG

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

    def test_save_restore_model_pretrained_inference_info(self):
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
        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(model, code_file, code_name)
        save_info = save_info_builder.build()

        model_id = self.save_recover_service.save_model(save_info)
        restored_model_info = self.save_recover_service.recover_model(model_id)
        self.assertTrue(model_equal(model, restored_model_info.model, imagenet_input))

    def test_save_restore_provenance_model_resnet18(self):
        model_name = resnet18.__name__
        self._test_save_restore_provenance_specific_model(model_name)

    def test_save_restore_provenance_model_resnet50(self):
        model_name = resnet50.__name__
        self._test_save_restore_provenance_specific_model(model_name)

    def test_save_restore_provenance_model_resnet152(self):
        model_name = resnet152.__name__
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
        # store model-0
        model = eval('{}(pretrained=True)'.format(model_name))
        if filename:
            code_file = MODEL_PATH.format(filename)
        else:
            code_file = MODEL_PATH.format(model_name)
        class_name = model_name
        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(model, code_file, class_name)
        save_info = save_info_builder.build()
        base_model_id = self.provenance_save_service.save_model(save_info)
        # -------------------------------------------------------------

        # store provenance-0
        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(code=code_file, model_class_name=class_name, base_model_id=base_model_id)

        imagenet_ts = ImagenetTrainService()
        self._add_imagenet_prov_state_dict(imagenet_ts, model)
        prov_train_serv_code = './inference_and_training/imagenet_train.py'
        prov_train_serv_class_name = 'ImagenetTrainService'
        prov_train_wrapper_code = './inference_and_training/imagenet_train.py'
        prov_train_wrapper_class_name = 'ImagenetTrainWrapper'
        raw_data = './data/reduced-custom-coco-data'
        prov_env = Environment({})
        train_kwargs = {'number_batches': 2}

        # TODO specify correct env, atm env is empty
        save_info_builder.add_prov_data(
            raw_data_path=raw_data, env=prov_env, train_service=imagenet_ts, train_kwargs=train_kwargs,
            code=prov_train_serv_code, class_name=prov_train_serv_class_name, wrapper_code=prov_train_wrapper_code,
            wrapper_class_name=prov_train_wrapper_class_name)
        save_info = save_info_builder.build()

        # save: train_state-0
        # it is a bit unintuitive but we have to store the prov data before training because through the training we
        # change the sate of the optimizer etc.
        model_id = self.provenance_save_service.save_model(save_info)
        print('modelId')
        print(model_id)
        # -------------------------------------------------------------

        # transitions model and train service:
        # model-0, train_state-0 -> # model-1, train_state-1
        imagenet_ts.train(model, **train_kwargs)
        self.recover_val_service.save_recover_val_info(model, model_id, dummy_input_shape=[10, 3, 300, 400])

        # "model" is in model_1
        # to recover model_1 we have saved train_state-0, and take it together with model_0
        recovered_model_info = self.provenance_save_service.recover_model(model_id)

        recovered_model_1 = recovered_model_info.model
        self.assertTrue(self.recover_val_service.check_recover_val(model_id, recovered_model_1))
        self.assertTrue(model_equal(model, recovered_model_1, imagenet_input))

        # save: train_state-1
        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(code=code_file, model_class_name=class_name, base_model_id=model_id)
        save_info_builder.add_prov_data(
            raw_data_path=raw_data, env=prov_env, train_service=imagenet_ts, train_kwargs=train_kwargs,
            code=prov_train_serv_code, class_name=prov_train_serv_class_name, wrapper_code=prov_train_wrapper_code,
            wrapper_class_name=prov_train_wrapper_class_name)
        save_info = save_info_builder.build()

        model_id_2 = self.provenance_save_service.save_model(save_info)
        print('modelId_2')
        print(model_id_2)
        # -------------------------------------------------------------

        # transitions model and train service:
        # model-1, train_state-1 -> # model-2, train_state-2
        imagenet_ts.train(model, **train_kwargs)
        self.recover_val_service.save_recover_val_info(model, model_id_2, dummy_input_shape=[10, 3, 300, 400])

        # "model" is in model_2
        # to recover model_2 we have saved train_state-1, and take it together with model_1
        recovered_model_info = self.provenance_save_service.recover_model(model_id_2)

        self.assertTrue(self.recover_val_service.check_recover_val(model_id_2, recovered_model_info.model))
        self.assertTrue(model_equal(model, recovered_model_info.model, imagenet_input))
        self.assertFalse(model_equal(recovered_model_1, recovered_model_info.model, imagenet_input))

        # check that restore of second model has not influenced restore of first model
        recovered_model_info = self.provenance_save_service.recover_model(model_id)

        recovered_model_1 = recovered_model_info.model
        self.assertTrue(self.recover_val_service.check_recover_val(model_id, recovered_model_info.model))

    def _add_imagenet_prov_state_dict(self, resnet_ts, model):
        set_deterministic()

        state_dict = {}

        optimizer = torch.optim.SGD(model.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)
        state_dict['optimizer'] = OptimizerWrapper(
            import_cmd='import torch',
            class_name='torch.optim.SGD',
            init_args={'lr': 1e-4, 'momentum': 0.9, 'weight_decay': 1e-4},
            config_args={},
            init_ref_type_args=['params'],
            instance=optimizer
        )

        data_wrapper = TrainCustomCoco('./data/reduced-custom-coco-data')
        state_dict['data'] = RestorableObjectWrapper(
            code='./networks/custom_coco.py',
            class_name='TrainCustomCoco',
            init_args={},
            config_args={'root': CURRENT_DATA_ROOT},
            init_ref_type_args=[],
            instance=data_wrapper
        )

        # Note use batch size 5 to reduce speed up tests
        dataloader = torch.utils.data.DataLoader(data_wrapper, batch_size=5, shuffle=False, num_workers=0,
                                                 pin_memory=True)
        state_dict['dataloader'] = RestorableObjectWrapper(
            import_cmd='from torch.utils.data import DataLoader',
            class_name='DataLoader',
            init_args={'batch_size': 5, 'shuffle': False, 'num_workers': 0, 'pin_memory': True},
            config_args={},
            init_ref_type_args=['dataset'],
            instance=dataloader
        )

        resnet_ts.state_objs = state_dict

    def test_save_restore_model_version(self):
        set_deterministic()
        model = resnet18()

        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(model, './networks/mynets/resnet18.py', 'resnet18')
        save_info = save_info_builder.build()
        model_id = self.save_recover_service.save_model(save_info)

        set_deterministic()
        model_version = resnet18()
        save_version_info_builder = ModelSaveInfoBuilder()
        save_version_info_builder.add_model_info(model_version, base_model_id=model_id)
        save_version_info = save_version_info_builder.build()

        model_version1_id = self.save_recover_service.save_model(save_version_info)

        model_version = resnet18(pretrained=True)
        save_version_info_builder = ModelSaveInfoBuilder()
        save_version_info_builder.add_model_info(model_version, base_model_id=model_version1_id)
        save_version_info = save_version_info_builder.build()
        model_version2_id = self.save_recover_service.save_model(save_version_info)

        restored_model_info = self.save_recover_service.recover_model(model_id)
        restored_model_info_version1 = self.save_recover_service.recover_model(model_version1_id)
        restored_model_info_version2 = self.save_recover_service.recover_model(model_version2_id)

        self.assertTrue(model_equal(model, restored_model_info.model, imagenet_input))
        self.assertTrue(model_equal(model, restored_model_info_version1.model, imagenet_input))
        self.assertTrue(model_equal(model_version, restored_model_info_version2.model, imagenet_input))
        self.assertFalse(
            model_equal(restored_model_info_version1.model, restored_model_info_version2.model, imagenet_input))

    def test_save_restore_model_and_recover_val(self):
        set_deterministic()
        model = resnet18()

        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(model, './networks/mynets/resnet18.py', 'resnet18')
        save_info = save_info_builder.build()

        model_id = self.save_recover_service.save_model(save_info)
        self.recover_val_service.save_recover_val_info(model, model_id, [10, 3, 300, 400])

        restored_model_info = self.save_recover_service.recover_model(model_id)

        self.assertTrue(self.recover_val_service.check_recover_val(model_id, restored_model_info.model))
        self.assertTrue(model_equal(model, restored_model_info.model, imagenet_input))

    def test_save_restore_model_version_and_recover_val(self):
        set_deterministic()
        model = resnet18()

        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(model, './networks/mynets/resnet18.py', 'resnet18')
        save_info = save_info_builder.build()
        model_id = self.save_recover_service.save_model(save_info)

        set_deterministic()
        model_version = resnet18()

        save_version_info_builder = ModelSaveInfoBuilder()
        save_version_info_builder.add_model_info(model_version, base_model_id=model_id)
        save_version_info = save_version_info_builder.build()
        model_version1_id = self.save_recover_service.save_model(save_version_info)
        self.recover_val_service.save_recover_val_info(model_version, model_version1_id, [10, 3, 300, 400])

        model_version = resnet18(pretrained=True)

        save_version_info_builder = ModelSaveInfoBuilder()
        save_version_info_builder.add_model_info(model_version, base_model_id=model_version1_id)
        save_version_info = save_version_info_builder.build()
        model_version2_id = self.save_recover_service.save_model(save_version_info)
        self.recover_val_service.save_recover_val_info(model_version, model_version2_id, [10, 3, 300, 400])

        restored_model_info = self.save_recover_service.recover_model(model_id)
        restored_model_version1_info = self.save_recover_service.recover_model(model_version1_id)
        restored_model_version2_info = self.save_recover_service.recover_model(model_version2_id)

        self.assertTrue(self.recover_val_service.check_recover_val(model_version1_id, restored_model_info.model))
        self.assertTrue(
            self.recover_val_service.check_recover_val(model_version2_id, restored_model_version2_info.model))
        self.assertTrue(model_equal(model, restored_model_info.model, imagenet_input))
        self.assertTrue(model_equal(model, restored_model_version1_info.model, imagenet_input))
        self.assertFalse(
            model_equal(restored_model_version1_info.model, restored_model_version2_info.model, imagenet_input))

    def test_model_save_size(self):
        model = resnet18(pretrained=True)
        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(model, './networks/mynets/resnet18.py', 'resnet18')
        save_info = save_info_builder.build()
        model_id = self.save_recover_service.save_model(save_info)

        model_id = model_id.replace(DICT, '')
        save_size = self.save_recover_service.model_save_size(model_id)

        # got from os (macOS finder info)
        code_file_size = 6802
        pickled_weights_size = 46837875

        model_info_size = self.mongo_service.document_size(ObjectId(model_id), MODEL_INFO)
        model_info_dict = self.mongo_service.get_dict(ObjectId(model_id), MODEL_INFO)
        restore_info_id = model_info_dict[RECOVER_INFO_ID].replace(DICT, '')
        restore_dict_size = self.mongo_service.document_size(ObjectId(restore_info_id), RECOVER_INFO)

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

        self.assertEqual(expected_size, save_size)

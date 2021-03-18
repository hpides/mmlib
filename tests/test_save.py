import os
import shutil
import unittest

import torch
from bson import ObjectId

from mmlib.deterministic import set_deterministic
from mmlib.equal import model_equal
from mmlib.persistence import MongoDictPersistenceService, FileSystemPersistenceService, DICT
from mmlib.save import BaselineSaveService, ProvenanceSaveService
from schema.environment import Environment
from schema.model_info import RECOVER_INFO_ID, MODEL_INFO_REPRESENT_TYPE
from schema.recover_info import RECOVER_INFO
from schema.recover_val import RECOVER_VAL
from schema.restorable_object import RestorableObjectWrapper, OptimizerWrapper
from schema.save_info_builder import ModelSaveInfoBuilder
from tests.inference_and_training.resnet_train import ResnetTrainService
from tests.networks.mynets.googlenet import googlenet
from tests.networks.mynets.mobilenet import mobilenet_v2
from tests.networks.mynets.resnet18 import resnet18
from util.dummy_data import imagenet_input
from util.mongo import MongoService

MONGO_CONTAINER_NAME = 'mongo-test'
COCO_ROOT = 'coco_root'
COCO_ANNOT = 'coco_annotations'

CONFIG = '/Users/nils/Studium/master-thesis/mmlib/tests/config.ini'


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

        os.environ['MMLIB_CONFIG'] = CONFIG

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

    def test_save_restore_provenance_model(self):

        # model-0, is "stored" - we just use pretrain to model the stored model
        model = resnet18(pretrained=True)
        code_file = './networks/mynets/{}.py'.format('resnet18')
        code_name = 'resnet18'

        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(model, code_file, code_name)

        resnet_ts = ResnetTrainService()
        self._add_resnet_prov_state_dict(resnet_ts, model)
        prov_train_serv_code = './inference_and_training/resnet_train.py'
        prov_train_serv_class_name = 'ResnetTrainService'
        prov_env = Environment({})
        train_kwargs = {'number_batches': 2}

        # TODO specify correct data path and env
        save_info_builder.add_prov_data(raw_data_path='data', env=prov_env, train_service=resnet_ts,
                                        train_kwargs=train_kwargs, code=prov_train_serv_code,
                                        class_name=prov_train_serv_class_name)
        save_info = save_info_builder.build()

        # save: train_state-0
        model_id = self.provenance_save_service.save_model(save_info)

        # transitions model and train service:
        # model-0, train_state-0 -> # model-1, train_state-1
        resnet_ts.train(model, **train_kwargs)

        # "model" is in model_1

        # to recover model_1 we hav saved train_state-0, and take it together with model_0 and the fixed command of
        # "resnet_ts.train(model, number_batches=2)" to produce model_1
        recovered_model_info = self.provenance_save_service.recover_model(model_id)

        self.assertTrue(model_equal(model, recovered_model_info.model, imagenet_input))

    def _add_resnet_prov_state_dict(self, resnet_ts, model):
        set_deterministic()

        # TODO think about how to get rid of magic strings
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

        data_wrapper = RestorableObjectWrapper(
            code='./networks/custom_coco.py',
            class_name='TrainCustomCoco',
            init_args={},
            # TODO think how to get data into here
            config_args={'root': 'coco_root', 'ann_file': 'coco_annotations'},
            init_ref_type_args=[]
        )
        # restore instance ein this way so that we do not have to read manually from config file
        data_wrapper.restore_instance()
        state_dict['data'] = data_wrapper

        dataloader = torch.utils.data.DataLoader(data_wrapper.instance, batch_size=64, shuffle=False, num_workers=0,
                                                 pin_memory=True)
        state_dict['dataloader'] = RestorableObjectWrapper(
            import_cmd='from torch.utils.data import DataLoader',
            class_name='DataLoader',
            init_args={'batch_size': 64, 'shuffle': False, 'num_workers': 0, 'pin_memory': True},
            config_args={},
            init_ref_type_args=['dataset'],
            instance=dataloader
        )

        resnet_ts.state_objs = state_dict

    # def test_save_restore_model_inf_info(self):
    #     file_names = ['mobilenet', 'resnet18']
    #     models = [mobilenet_v2, resnet18]
    #     for file_name, model in zip(file_names, models):
    #         code_name = model.__name__
    #         model = model()
    #         code_file = './networks/mynets/{}.py'.format(file_name)
    #
    #         self._test_save_restore_model_inference_info(code_file, code_name, model)

    # def _test_save_restore_model_inference_info(self, code_file, code_name, model):
    #     save_info_builder = ModelSaveInfoBuilder()
    #     save_info_builder.add_model_info(model, code_file, code_name)
    #     save_info_builder.add_recover_val(dummy_input_shape=[10, 3, 300, 400])
    #     data_wrapper = RestorableObjectWrapper(
    #         code='./networks/custom_coco.py',
    #         class_name='InferenceCustomCoco',
    #         init_args={},
    #         config_args={'root': COCO_ROOT, 'ann_file': COCO_ANNOT},
    #         init_ref_type_args=[]
    #     )
    #     dataloader = RestorableObjectWrapper(
    #         import_cmd='from torch.utils.data import DataLoader',
    #         class_name='DataLoader',
    #         init_args={'batch_size': 64, 'shuffle': False, 'num_workers': 0, 'pin_memory': True},
    #         config_args={},
    #         init_ref_type_args=['dataset']
    #     )
    #     preprocessor = RestorableObjectWrapper(
    #         code='./networks/dummy_preprocessor.py',
    #         class_name='DummyPreprocessor',
    #         init_args={},
    #         config_args={},
    #         init_ref_type_args=[]
    #     )
    #     environment = Environment(environment_data={'cpu': 'test'})
    #     save_info_builder.add_inference_info(data_wrapper, dataloader, preprocessor, environment)
    #     save_info = save_info_builder.build()
    #
    #     model_id = self.save_recover_service.save_model(save_info)
    #     restored_model_info = self.save_recover_service.recover_model(model_id, inference_info=True)
    #
    #     self.assertTrue(restored_model_info.inference_info.data_wrapper.instance)
    #     self.assertTrue(restored_model_info.inference_info.dataloader.instance)
    #     self.assertTrue(restored_model_info.inference_info.pre_processor.instance)
    #     self.assertTrue(restored_model_info.inference_info.environment)
    #
    #     self.assertTrue(model_equal(model, restored_model_info.model, imagenet_input))

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
        self.assertFalse(
            model_equal(restored_model_info_version1.model, restored_model_info_version2.model, imagenet_input))

    def test_save_restore_model_and_recover_val(self):
        set_deterministic()
        model = resnet18()

        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(model, './networks/mynets/resnet18.py', 'resnet18')
        save_info_builder.add_recover_val(dummy_input_shape=[10, 3, 300, 400])
        save_info = save_info_builder.build()

        model_id = self.save_recover_service.save_model(save_info)

        restored_model_info = self.save_recover_service.recover_model(model_id, check_recover_val=True)

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
        save_version_info_builder.add_recover_val(dummy_input_shape=[10, 3, 300, 400])
        save_version_info = save_version_info_builder.build()
        model_version1_id = self.save_recover_service.save_model(save_version_info)

        model_version = resnet18(pretrained=True)

        save_version_info_builder = ModelSaveInfoBuilder()
        save_version_info_builder.add_model_info(model_version, base_model_id=model_version1_id)
        save_version_info_builder.add_recover_val(dummy_input_shape=[10, 3, 300, 400])
        save_version_info = save_version_info_builder.build()
        model_version2_id = self.save_recover_service.save_model(save_version_info)

        restored_model_info = self.save_recover_service.recover_model(model_id)
        restored_model_version1_info = self.save_recover_service.recover_model(model_version1_id,
                                                                               check_recover_val=True)
        restored_model_version2_info = self.save_recover_service.recover_model(model_version2_id,
                                                                               check_recover_val=True)

        self.assertTrue(model_equal(model, restored_model_info.model, imagenet_input))
        self.assertTrue(model_equal(model, restored_model_version1_info.model, imagenet_input))
        self.assertFalse(
            model_equal(restored_model_version1_info.model, restored_model_version2_info.model, imagenet_input))

    def test_model_save_size(self):
        self._test_model_save_size()

    def test_model_save_size_recover_val(self):
        self._test_model_save_size(recover_val=True)

    def _test_model_save_size(self, recover_val=False):
        model = resnet18(pretrained=True)
        if recover_val:
            save_info_builder = ModelSaveInfoBuilder()
            save_info_builder.add_model_info(model, './networks/mynets/resnet18.py', 'resnet18')
            save_info_builder.add_recover_val(dummy_input_shape=[10, 3, 300, 400])
            save_info = save_info_builder.build()
            model_id = self.save_recover_service.save_model(save_info)
        else:
            save_info_builder = ModelSaveInfoBuilder()
            save_info_builder.add_model_info(model, './networks/mynets/resnet18.py', 'resnet18')
            save_info = save_info_builder.build()
            model_id = self.save_recover_service.save_model(save_info)

        model_id = model_id.replace(DICT, '')
        save_size = self.save_recover_service.model_save_size(model_id)

        # got from os (macOS finder info)
        code_file_size = 6802
        pickled_weights_size = 46837875

        model_info_size = self.mongo_service.document_size(ObjectId(model_id), MODEL_INFO_REPRESENT_TYPE)
        model_info_dict = self.mongo_service.get_dict(ObjectId(model_id), MODEL_INFO_REPRESENT_TYPE)
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

        if recover_val:
            restore_info = self.mongo_service.get_dict(ObjectId(restore_info_id), RECOVER_INFO)
            recover_val_id = restore_info[RECOVER_VAL].replace(DICT, '')
            val_dict_size = self.mongo_service.document_size(ObjectId(recover_val_id), RECOVER_VAL)
            expected_size += val_dict_size

        self.assertEqual(expected_size, save_size)

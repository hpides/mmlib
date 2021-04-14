import os
import shutil
import unittest

import torch

from mmlib.constants import CURRENT_DATA_ROOT, MMLIB_CONFIG
from mmlib.deterministic import set_deterministic
from mmlib.equal import model_equal
from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from mmlib.save import ProvenanceSaveService
from mmlib.track_env import track_current_environment
from schema.restorable_object import OptimizerWrapper, RestorableObjectWrapper
from schema.save_info_builder import ModelSaveInfoBuilder
from tests.inference_and_training.imagenet_train import ImagenetTrainService
from tests.networks.custom_coco import TrainCustomCoco
from tests.networks.mynets.mobilenet import mobilenet_v2
from tests.networks.mynets.resnet18 import resnet18
from tests.test_baseline_save_servcie import MONGO_CONTAINER_NAME
from util.dummy_data import imagenet_input
from util.mongo import MongoService

MODEL_PATH = './networks/mynets/{}.py'
CONFIG = './local-config.ini'


class TestProvSaveService(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_path = './filesystem-tmp'
        self.abs_tmp_path = os.path.abspath(self.tmp_path)

        self.__clean_up()
        # run mongo DB locally in docker container
        os.system('docker run --rm --name %s -it -p 27017:27017 -d  mongo:4.4.3 ' % MONGO_CONTAINER_NAME)

        self.mongo_service = MongoService('127.0.0.1', 'mmlib')
        file_pers_service = FileSystemPersistenceService(self.tmp_path)
        dict_pers_service = MongoDictPersistenceService()
        self.provenance_save_service = ProvenanceSaveService(file_pers_service, dict_pers_service)

        os.mkdir(self.abs_tmp_path)

        os.environ[MMLIB_CONFIG] = CONFIG

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
        ###############################################################################
        # as a first step we store model model-0
        ###############################################################################
        # it will be stored as a full model since there is no model it was derived from
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

        ################################################################################################################
        # as next we define the provenance data, that can not be automatically inferred
        # All this information has to be only once, for every future version the same prov data can be used
        # exceptions are the train_kwargs and the raw_data
        ################################################################################################################
        # define what train service will be used to train the model, in our case the ImagenetTrainService (inherits
        # from the abstract class TrainService)
        prov_train_serv_code = './inference_and_training/imagenet_train.py'
        prov_train_serv_class_name = 'ImagenetTrainService'
        # define the train wrapper, in our case we use the ImagenetTrainWrapper (inherits from the abstract class
        # TrainService)
        prov_train_wrapper_code = './inference_and_training/imagenet_train.py'
        prov_train_wrapper_class_name = 'ImagenetTrainWrapper'
        # we also have to track the current environment, to store it later
        prov_env = track_current_environment()
        # as a last step we have to define the data that should be used and how the train method should be parametrized
        raw_data = './data/reduced-custom-coco-data'
        train_kwargs = {'number_batches': 2}

        # to train the model we use the imagenet train service specified above
        imagenet_ts = ImagenetTrainService()
        # to make the train service work we have to initialize its state dict containing objects that are required for
        # training, for example: optimizer and dataloader. The objects in the state dict have to be of type
        # RestorableObjectWrapper so that the hold instances and their state can be stored and restored

        # set deterministic for debugging purposes
        set_deterministic()

        state_dict = {}

        # before we can define the data loader, we have to define the data wrapper
        # for this test case we will use the data from our custom coco dataset
        data_wrapper = TrainCustomCoco(raw_data)
        state_dict['data'] = RestorableObjectWrapper(
            code='./networks/custom_coco.py',
            class_name='TrainCustomCoco',
            init_args={},
            config_args={'root': CURRENT_DATA_ROOT},
            init_ref_type_args=[],
            instance=data_wrapper
        )

        # as a dataloader we use the standard implementation provided by pyTorch
        # this is why we instead of specifying the code path, we specify an import cmd
        # also we to track all arguments that have been used for initialization of this objects
        batch_size = 5
        shuffle = False
        num_workers = 0
        pin_memory = True
        dataloader = torch.utils.data.DataLoader(data_wrapper, batch_size=batch_size, shuffle=shuffle,
                                                 num_workers=num_workers, pin_memory=pin_memory)
        state_dict['dataloader'] = RestorableObjectWrapper(
            import_cmd='from torch.utils.data import DataLoader',
            class_name='DataLoader',
            init_args={'batch_size': batch_size, 'shuffle': pin_memory, 'num_workers': num_workers,
                       'pin_memory': pin_memory},
            config_args={},
            init_ref_type_args=['dataset'],
            instance=dataloader
        )

        # while the other objects do not have an internal state (other than the basic parameters) an optimizer can have
        # some more extensive state. In pyTorch it offers also the method .state_dict(). To store and restore we use an
        # Optimizer wrapper object.
        lr = 1e-4
        momentum = 0.9
        weight_decay = 1e-4
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        state_dict['optimizer'] = OptimizerWrapper(
            import_cmd='import torch',
            class_name='torch.optim.SGD',
            init_args={'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay},
            config_args={},
            init_ref_type_args=['params'],
            instance=optimizer
        )

        # having created all the objects needed for imagenet training we can plug the state dict into the train servcie
        imagenet_ts.state_objs = state_dict

        ################################################################################################################
        # having specified all the provenance information that will be used to train a model, we can store it
        ################################################################################################################
        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(code=code_file, model_class_name=class_name, base_model_id=base_model_id)
        save_info_builder.add_prov_data(
            raw_data_path=raw_data, env=prov_env, train_service=imagenet_ts, train_kwargs=train_kwargs,
            code=prov_train_serv_code, class_name=prov_train_serv_class_name, wrapper_code=prov_train_wrapper_code,
            wrapper_class_name=prov_train_wrapper_class_name)
        save_info = save_info_builder.build()

        # restoring this model will result in a model that was trained according to the given provenance data
        # in this case it should be equivalent to the initial model trained using the specified train_service using the
        # specified data and train kwargs
        model_id = self.provenance_save_service.save_model(save_info)

        imagenet_ts.train(model, **train_kwargs)
        recovered_model_info = self.provenance_save_service.recover_model(model_id)
        self.assertTrue(model_equal(model, recovered_model_info.model, imagenet_input))

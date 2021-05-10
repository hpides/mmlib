import os

import torch
from torch.utils.data import DataLoader

from mmlib.constants import CURRENT_DATA_ROOT, MMLIB_CONFIG
from mmlib.deterministic import set_deterministic
from mmlib.equal import model_equal
from mmlib.save import ProvenanceSaveService
from mmlib.track_env import track_current_environment
from schema.restorable_object import OptimizerWrapper, RestorableObjectWrapper
from schema.save_info_builder import ModelSaveInfoBuilder
from tests.example_files.data.custom_coco import TrainCustomCoco
from tests.example_files.imagenet_train import ImagenetTrainService, OPTIMIZER, DATALOADER, DATA, ImagenetTrainWrapper
from tests.example_files.mynets.mobilenet import mobilenet_v2
from tests.example_files.mynets.resnet18 import resnet18
from tests.save.test_baseline_save_servcie import TestBaselineSaveService
from util.dummy_data import imagenet_input

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(FILE_PATH, '../example_files/mynets/{}.py')
CONFIG = os.path.join(FILE_PATH, '../example_files/local-config.ini')


class TestProvSaveService(TestBaselineSaveService):

    def setUp(self) -> None:
        super().setUp()
        assert os.path.isfile(CONFIG), \
            'to run these tests define your onw config file named \'local-config\' with respect to the template file'
        os.environ[MMLIB_CONFIG] = CONFIG

    def init_save_service(self, dict_pers_service, file_pers_service):
        self.save_service = ProvenanceSaveService(file_pers_service, dict_pers_service)

    def test_save_restore_provenance_model_resnet18(self):
        model = resnet18(pretrained=True)
        self._test_save_restore_provenance_specific_model(model)

    def test_save_restore_provenance_model_mobilenet(self):
        model = mobilenet_v2(pretrained=True)
        self._test_save_restore_provenance_specific_model(model)

    # googlenet has some problems when restored form state_dict with aux loss
    # NOTE think about not using googlenet for experiments
    # def test_save_restore_provenance_model_googlenet(self):
    #     model_name = googlenet.__name__
    #     self._test_save_restore_provenance_specific_model(model_name)

    def _test_save_restore_provenance_specific_model(self, model):
        ###############################################################################
        # as a first step we store model model-0
        ###############################################################################
        # it will be stored as a full model since there is no model it was derived from
        save_info_builder = ModelSaveInfoBuilder()
        env = track_current_environment()
        save_info_builder.add_model_info(model=model, env=env)
        save_info = save_info_builder.build()

        base_model_id = self.save_service.save_model(save_info)

        ################################################################################################################
        # as next we define the provenance data, that can not be automatically inferred
        # All this information has to be only once, for every future version the same prov data can be used
        # exceptions are the train_kwargs and the raw_data
        ################################################################################################################
        # we also have to track the current environment, to store it later
        prov_env = track_current_environment()
        # as a last step we have to define the data that should be used and how the train method should be parametrized
        raw_data = os.path.join(FILE_PATH, '../example_files/data/reduced-custom-coco-data')
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
        state_dict[DATA] = RestorableObjectWrapper(
            config_args={'root': CURRENT_DATA_ROOT},
            instance=data_wrapper
        )

        # as a dataloader we use the standard implementation provided by pyTorch
        # this is why we instead of specifying the code path, we specify an import cmd
        # also we to track all arguments that have been used for initialization of this objects
        data_loader_kwargs = {'batch_size': 5, 'shuffle': True, 'num_workers': 0, 'pin_memory': True}
        dataloader = DataLoader(data_wrapper, **data_loader_kwargs)
        state_dict[DATALOADER] = RestorableObjectWrapper(
            import_cmd='from torch.utils.data import DataLoader',
            init_args=data_loader_kwargs,
            init_ref_type_args=['dataset'],
            instance=dataloader
        )

        # while the other objects do not have an internal state (other than the basic parameters) an optimizer can have
        # some more extensive state. In pyTorch it offers also the method .state_dict(). To store and restore we use an
        # Optimizer wrapper object.
        optimizer_kwargs = {'lr': 1e-4, 'momentum': 0.9, 'weight_decay': 1e-4}
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_kwargs)
        state_dict[OPTIMIZER] = OptimizerWrapper(
            import_cmd='from torch.optim import SGD',
            init_args=optimizer_kwargs,
            init_ref_type_args=['params'],
            instance=optimizer
        )

        # having created all the objects needed for imagenet training we can plug the state dict into the train servcie
        imagenet_ts.state_objs = state_dict

        # finally we wrap the train service in the corresponding wrapper
        ts_wrapper = ImagenetTrainWrapper(instance=imagenet_ts)
        ################################################################################################################
        # having specified all the provenance information that will be used to train a model, we can store it
        ################################################################################################################
        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(base_model_id=base_model_id, env=prov_env)
        save_info_builder.add_prov_data(
            raw_data_path=raw_data, train_kwargs=train_kwargs, train_service_wrapper=ts_wrapper)
        save_info = save_info_builder.build()

        ################################################################################################################
        # restoring this model will result in a model that was trained according to the given provenance data
        # in this case it should be equivalent to the initial model trained using the specified train_service using the
        # specified data and train kwargs
        ################################################################################################################
        model_id = self.save_service.save_model(save_info)

        imagenet_ts.train(model, **train_kwargs)
        self.save_service.add_weights_hash_info(model_id, model)
        recovered_model_info = self.save_service.recover_model(model_id)
        recovered_model_1 = recovered_model_info.model

        ################################################################################################################
        # Having defined the provenance information above storing a second version is a lot shorter
        ################################################################################################################
        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(base_model_id=model_id, env=prov_env)
        save_info_builder.add_prov_data(raw_data_path=raw_data, train_kwargs=train_kwargs,
                                        train_service_wrapper=ts_wrapper)
        save_info = save_info_builder.build()

        model_id_2 = self.save_service.save_model(save_info)

        imagenet_ts.train(model, **train_kwargs)
        self.save_service.add_weights_hash_info(model_id_2, model)

        recovered_model_info = self.save_service.recover_model(model_id_2)
        self.assertTrue(model_equal(model, recovered_model_info.model, imagenet_input))
        self.assertFalse(model_equal(recovered_model_1, recovered_model_info.model, imagenet_input))

import os

import torch
from torch.utils.data import DataLoader

from mmlib.constants import CURRENT_DATA_ROOT, MMLIB_CONFIG
from mmlib.deterministic import set_deterministic
from mmlib.equal import model_equal
from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from mmlib.save import ProvenanceSaveService
from mmlib.track_env import track_current_environment
from schema.restorable_object import RestorableObjectWrapper, OptimizerWrapper
from schema.save_info_builder import ModelSaveInfoBuilder
from tests.example_files.data.custom_coco import TrainCustomCoco
from tests.example_files.imagenet_train import ImagenetTrainService, DATALOADER, OPTIMIZER, ImagenetTrainWrapper, DATA
from tests.example_files.mynets.resnet18 import resnet18
from util.dummy_data import imagenet_input

CONTAINER_NAME = 'mongo-test'
TARGET_FILE_SYSTEM_DIR = 'filesystem-tmp'


def initi_train_service():
    global state_dict, imagenet_ts
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
    # having created all the objects needed for imagenet training we can plug the state dict into the train service
    imagenet_ts.state_objs = state_dict


if __name__ == '__main__':
    # initialize a service to store files
    os.environ[MMLIB_CONFIG] = '../tests/example_files/local-config.ini'
    abs_tmp_path = os.path.abspath(TARGET_FILE_SYSTEM_DIR)
    file_pers_service = FileSystemPersistenceService(abs_tmp_path)
    # run mongoDB locally in docker container and initialize service to store dictionaries (JSON)
    os.system('docker kill %s' % CONTAINER_NAME)
    os.system('docker run --rm --name %s -it -p 27017:27017 -d  mongo:4.4.3 ' % CONTAINER_NAME)
    dict_pers_service = MongoDictPersistenceService()
    # initialize baseline save service
    save_service = ProvenanceSaveService(file_pers_service, dict_pers_service)
    # save the first version of the model
    model = resnet18(pretrained=True)
    save_info_builder = ModelSaveInfoBuilder()
    save_info_builder.add_model_info(model=model)
    save_info = save_info_builder.build()
    base_model_id = save_service.save_model(save_info)

    # create train service
    imagenet_ts = ImagenetTrainService()
    raw_data = '../tests/example_files/data/reduced-custom-coco-data'
    initi_train_service()
    # specify the provenance information
    prov_env = track_current_environment()
    # as a last step we have to define the data that should be used and how the train method should be parametrized
    train_kwargs = {'number_batches': 2}
    # wrap the train service in the corresponding wrapper
    ts_wrapper = ImagenetTrainWrapper(instance=imagenet_ts)
    # save the model
    save_info_builder = ModelSaveInfoBuilder()
    save_info_builder.add_model_info(base_model_id=base_model_id)
    save_info_builder.add_prov_data(
        raw_data_path=raw_data, env=prov_env, train_kwargs=train_kwargs, train_service_wrapper=ts_wrapper)
    save_info = save_info_builder.build()
    model_id = save_service.save_model(save_info)

    imagenet_ts.train(model, **train_kwargs)
    recovered_model_info = save_service.recover_model(model_id)
    recovered_model = recovered_model_info.model
    if model_equal(model, recovered_model, imagenet_input):
        print('Success: the stored and the restored models are equal!')

    # kill the docker container
    os.system('docker kill %s' % CONTAINER_NAME)

import os
import tempfile

import torch

from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from schema.restorable_object import RestorableObjectWrapper
from tests.dummy_code.resnet_train import OptimizerWrapper, ResnetTrainService, ResnetTrainWrapper
from tests.networks.mynets.resnet18 import resnet18
from tests.test_dict_persistence import MONGO_CONTAINER_NAME
from tests.test_save import CONFIG

if __name__ == '__main__':
    # set path for config file
    os.environ['MMLIB_CONFIG'] = CONFIG

    # open a tmp path for filesystem pers service
    with tempfile.TemporaryDirectory() as tmp_path:
        file_ps = FileSystemPersistenceService(base_path=tmp_path)

        # start mongoDB for mongodict pers service
        os.system('docker run --rm --name %s -it -p 27017:27017 -d  mongo:4.4.3 ' % MONGO_CONTAINER_NAME)
        dict_ps = MongoDictPersistenceService()

        model = resnet18()

        # prepare state dict for ResnetTrainService
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
            code='../networks/custom_coco.py',
            class_name='InferenceCustomCoco',
            init_args={},
            config_args={'root': 'coco_root', 'ann_file': 'coco_annotations'},
            init_ref_type_args=[]
        )
        # restore instance ein this way so that we do not have to read manually from config file
        data_wrapper.restore_instance()
        state_dict['data'] = data_wrapper

        dataloader = torch.utils.data.DataLoader(data_wrapper.instance, batch_size=64, shuffle=True, num_workers=0,
                                                 pin_memory=True)
        state_dict['dataloader'] = RestorableObjectWrapper(
            import_cmd='from torch.utils.data import DataLoader',
            class_name='DataLoader',
            init_args={'batch_size': 64, 'shuffle': False, 'num_workers': 0, 'pin_memory': True},
            config_args={},
            init_ref_type_args=['dataset'],
            instance=dataloader
        )

        # create ResnetTrainService and plug in the state dict
        resnet_ts = ResnetTrainService()
        resnet_ts.state_objs = state_dict

        # wrap the resnet_ts in a wrapper to be restorable on other devices and persist teh wrapper
        ts_wrapper = ResnetTrainWrapper(
            code='./resnet_train.py',
            class_name='ResnetTrainService',
            instance=resnet_ts
        )

        # train with 2 batches
        resnet_ts.train(model, number_batches=2)

        # persist the train service
        ts_wrapper_id = ts_wrapper.persist(file_ps, dict_ps)

        # load the wrapper back to continue training
        ts_wrapper_new = ResnetTrainWrapper.load(ts_wrapper_id, file_ps, dict_ps, tmp_path)
        ts_wrapper_new.restore_instance(file_ps, dict_ps, tmp_path)
        resnet_ts_new: ResnetTrainService = ts_wrapper_new.instance

        # train with another 2 batches
        resnet_ts_new.train(model, number_batches=2)

        #
        # resnet_ts_new: ResnetTrainService = ts_wrapper_new.instance
        #
        # # use the restored train service to
        #
        # resnet_ts_new.train(model, number_batches=2)
        #
        # nid = ts_wrapper_new.persist(file_ps, dict_ps)
        # nw = ResnetTrainWrapper.load(nid, file_ps, dict_ps, tmp_path)
        # nw.restore_instance(file_ps, dict_ps, tmp_path)
        # new: ResnetTrainService = ts_wrapper_new.instance
        #
        # ts_new.train(model, number_batches=2)
        #
        # print('test')

        #
        # optimizer_id = state_dict['optimizer'].persist(file_ps, dict_ps)
        #
        # optimizer_wrapper = RestorableObjectWrapper.load(optimizer_id, file_ps, dict_ps, tmp_path)
        # optimizer_wrapper.restore_instance(ref_type_args={'params': model.parameters()})
        # optimizer = optimizer_wrapper.instance
        #
        # data_wrapper_id = state_dict['data'].persist(file_ps, dict_ps)
        # data_wrapper = RestorableObjectWrapper.load(data_wrapper_id, file_ps, dict_ps, tmp_path)
        # data_wrapper.restore_instance()
        # data = data_wrapper.instance
        #
        # dataloader_wrapper_id = state_dict['dataloader'].persist(file_ps, dict_ps)
        # dataloader_wrapper = RestorableObjectWrapper.load(dataloader_wrapper_id, file_ps, dict_ps, tmp_path)
        # dataloader_wrapper.restore_instance({'dataset': data})
        # dataloader = dataloader_wrapper.instance
        #
        # print(optimizer_id)

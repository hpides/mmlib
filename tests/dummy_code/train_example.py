import os
import tempfile

import torch

from mmlib.equal import model_equal
from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from schema.restorable_object import RestorableObjectWrapper
from tests.dummy_code.resnet_train import OptimizerWrapper, ResnetTrainService, ResnetTrainWrapper
from tests.networks.mynets.resnet18 import resnet18
from tests.test_dict_persistence import MONGO_CONTAINER_NAME
from tests.test_save import CONFIG
from util.dummy_data import imagenet_input

if __name__ == '__main__':
    # set path for config file
    os.environ['MMLIB_CONFIG'] = CONFIG

    # open a tmp path for filesystem pers service
    with tempfile.TemporaryDirectory() as tmp_path:
        file_ps = FileSystemPersistenceService(base_path=tmp_path)

        # start mongoDB for mongodict pers service
        os.system('docker run --rm --name %s -it -p 27017:27017 -d  mongo:4.4.3 ' % MONGO_CONTAINER_NAME)
        dict_ps = MongoDictPersistenceService()

        model = resnet18(pretrained=True)

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

        # save the model for later comparison
        resnet_weights_path = os.path.join(tmp_path, 'res18_model_weights')
        torch.save(model.state_dict(), resnet_weights_path)

        # persist the train service
        ts_wrapper_id = ts_wrapper.persist(file_ps, dict_ps)

        # load the wrapper back to continue training
        ts_wrapper_new = ResnetTrainWrapper.load(ts_wrapper_id, file_ps, dict_ps, tmp_path)
        ts_wrapper_new.restore_instance(file_ps, dict_ps, tmp_path)
        resnet_ts_new: ResnetTrainService = ts_wrapper_new.instance

        # train with another 2 batches
        resnet_ts_new.train(model, number_batches=2)

        # restore model from before and restore ResnetTrainWrapper to compare results
        # we expect the resulting models to be the same
        second_model = resnet18()
        second_model.load_state_dict(torch.load(resnet_weights_path))

        # load the same wrapper back to continue second (identical) training
        ts_wrapper_new = ResnetTrainWrapper.load(ts_wrapper_id, file_ps, dict_ps, tmp_path)
        ts_wrapper_new.restore_instance(file_ps, dict_ps, tmp_path)
        resnet_ts_new: ResnetTrainService = ts_wrapper_new.instance

        # train with another 2 batches
        resnet_ts_new.train(second_model, number_batches=2)

        # starting from the same intermediate model the both restored versions of the resnet train service
        # should train the models in the same way
        # TODO check for other models and on GPU where we have to set "set.deterministic()"
        eq = model_equal(model, second_model, imagenet_input)

        print(eq)

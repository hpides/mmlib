import os
import os
import tempfile

import torch

from mmlib.persistence import AbstractDictPersistenceService, AbstractFilePersistenceService, \
    FileSystemPersistenceService, MongoDictPersistenceService
from schema.restorable_object import StateDictRestorableObjectWrapper, RESTORABLE_OBJECT, STATE_DICT, \
    RestorableObjectWrapper, TrainService
from tests.networks.mynets.resnet18 import resnet18
from tests.test_dict_persistence import MONGO_CONTAINER_NAME
from tests.test_save import CONFIG
from util.init_from_file import create_object


class ResnetTrainService(TrainService):
    def train(self, model: torch.nn.Module, number_batches=None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        train_loader = self._get_dataloader()

        model.to(device)

        # switch to train mode
        model.train()

        for i, (images, target) in enumerate(train_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = torch.nn.CrossEntropyLoss()(output, target)

            # compute gradient and do SGD step
            # TODO restore manually here since optimizer needs model params
            optimizer = self._get_optimizer(model.parameters())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not (number_batches is None or i < number_batches - 1):
                break

    def _get_dataloader(self):
        dataloader = self.state_objs['dataloader'].instance
        return dataloader

    def _get_optimizer(self, parameters):
        optimizer_wrapper = self.state_objs['optimizer']
        optimizer_wrapper.restore_instance({'params': parameters})
        return optimizer_wrapper.instance


class ResnetTrainWrapper(StateDictRestorableObjectWrapper):

    def restore_instance(self, file_pers_service: AbstractFilePersistenceService,
                         dict_pers_service: AbstractDictPersistenceService, restore_root: str):
        state_dict = {}

        restored_dict = dict_pers_service.recover_dict(self.store_id, RESTORABLE_OBJECT)
        state_objs = restored_dict[STATE_DICT]

        # NOTE: Dataloader instance is loaded in the train routine
        state_dict['optimizer'] = RestorableObjectWrapper.load(
            state_objs['optimizer'], file_pers_service, dict_pers_service, restore_root)

        data_wrapper = RestorableObjectWrapper.load(
            state_objs['data'], file_pers_service, dict_pers_service, restore_root)
        state_dict['data'] = data_wrapper
        data_wrapper.restore_instance()

        dataloader = RestorableObjectWrapper.load(
            state_objs['dataloader'], file_pers_service, dict_pers_service, restore_root)
        state_dict['dataloader'] = dataloader
        dataloader.restore_instance(ref_type_args={'dataset': data_wrapper.instance})

        self.instance = create_object(code=self.code, class_name=self.class_name)
        self.instance.state_objs = state_dict


if __name__ == '__main__':
    # init dict with internal state here

    os.environ['MMLIB_CONFIG'] = CONFIG

    with tempfile.TemporaryDirectory() as tmp_path:
        file_ps = FileSystemPersistenceService(base_path=tmp_path)
        os.system('docker run --rm --name %s -it -p 27017:27017 -d  mongo:4.4.3 ' % MONGO_CONTAINER_NAME)
        dict_ps = MongoDictPersistenceService()

        state_dict = {}
        # torch.optim.SGD(model.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)
        # TODO implement state_dict save for optimizer
        state_dict['optimizer'] = RestorableObjectWrapper(
            import_cmd='import torch',
            class_name='torch.optim.SGD',
            init_args={'lr': 1e-4, 'momentum': 0.9, 'weight_decay': 1e-4},
            config_args={},
            init_ref_type_args=['params']
        )

        state_dict['data'] = RestorableObjectWrapper(
            code='../networks/custom_coco.py',
            class_name='InferenceCustomCoco',
            init_args={},
            config_args={'root': 'coco_root', 'ann_file': 'coco_annotations'},
            init_ref_type_args=[]
        )

        state_dict['dataloader'] = RestorableObjectWrapper(
            import_cmd='from torch.utils.data import DataLoader',
            class_name='DataLoader',
            init_args={'batch_size': 64, 'shuffle': False, 'num_workers': 0, 'pin_memory': True},
            config_args={},
            init_ref_type_args=['dataset']
        )

        ts = ResnetTrainService()
        ts.state_objs = state_dict

        ts_wrapper = ResnetTrainWrapper(
            code='./train.py',
            class_name='ResnetTrainService',
            instance=ts
        )

        ts_wrapper_id = ts_wrapper.persist(file_ps, dict_ps)

        ts_wrapper_new = ResnetTrainWrapper.load(ts_wrapper_id, file_ps, dict_ps, tmp_path)
        ts_wrapper_new.restore_instance(file_ps, dict_ps, tmp_path)
        ts_new: ResnetTrainService = ts_wrapper_new.instance

        model = resnet18()
        ts_new.train(model, number_batches=2)

        print('test')

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

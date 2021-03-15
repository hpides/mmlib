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


class ResnetTrainService(TrainService):
    def train(self, model: torch.nn.Module):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        train_loader = self._get_dataloader()

        model.to(device)

        # switch to train mode
        model.train()

        outputs = []

        for i, (images, target) in enumerate(train_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = torch.nn.CrossEntropyLoss()(output, target)

            # compute gradient and do SGD step
            optimizer = self._get_optimizer()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def _get_dataloader(self):
        dataloader = self.state_objs['data_loader'].instace
        return dataloader

    def _get_optimizer(self):
        optimizer = self.state_objs['optimizer'].instace
        return optimizer


class ResnetTrainWrapper(StateDictRestorableObjectWrapper):

    def restore_instance(self, file_pers_service: AbstractFilePersistenceService,
                         dict_pers_service: AbstractDictPersistenceService, restore_root: str):
        state_dict = {}

        restored_dict = dict_pers_service.recover_dict(self.store_id, RESTORABLE_OBJECT)
        state_objs = restored_dict[STATE_DICT]

        optimizer = RestorableObjectWrapper.load(state_objs['optimizer'], file_pers_service, dict_pers_service,
                                                 restore_root)
        state_dict['optimizer'] = optimizer  # TODO fix can only be initialized in the train method

        dataloader = RestorableObjectWrapper.load(state_objs['data_loader'], file_pers_service, dict_pers_service,
                                                  restore_root)
        state_dict['data_loader'] = dataloader


if __name__ == '__main__':
    # init dict with internal state here

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

        model = resnet18()

        optimizer_id = state_dict['optimizer'].persist(file_ps, dict_ps)

        optimizer_wrapper = RestorableObjectWrapper.load(optimizer_id, file_ps, dict_ps, tmp_path)
        optimizer_wrapper.restore_instance(ref_type_args={'params': model.parameters()})
        optimizer = optimizer_wrapper.instance

        state_dict['data'] = RestorableObjectWrapper(
            code='../networks/custom_coco.py',
            class_name='InferenceCustomCoco',
            init_args={},
            config_args={'root': 'coco_root', 'ann_file': 'coco_annotations'},
            init_ref_type_args=[]
        )

        os.environ['MMLIB_CONFIG'] = CONFIG
        data_wrapper_id = state_dict['data'].persist(file_ps, dict_ps)
        data_wrapper = RestorableObjectWrapper.load(data_wrapper_id, file_ps, dict_ps, tmp_path)
        data_wrapper.restore_instance()
        data = data_wrapper.instance

        state_dict['dataloader'] = RestorableObjectWrapper(
            import_cmd='from torch.utils.data import DataLoader',
            class_name='DataLoader',
            init_args={'batch_size': 64, 'shuffle': False, 'num_workers': 0, 'pin_memory': True},
            config_args={},
            init_ref_type_args=['dataset']
        )
        dataloader_wrapper_id = state_dict['dataloader'].persist(file_ps, dict_ps)
        dataloader_wrapper = RestorableObjectWrapper.load(dataloader_wrapper_id, file_ps, dict_ps, tmp_path)
        dataloader_wrapper.restore_instance({'dataset': data})
        dataloader = dataloader_wrapper.instance


        print(optimizer_id)

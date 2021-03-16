import torch

from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from schema.restorable_object import TrainService, StateDictRestorableObjectWrapper, RESTORABLE_OBJECT, STATE_DICT, \
    RestorableObjectWrapper, StateFileRestorableObjectWrapper
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
        optimizer_wrapper: OptimizerWrapper = self.state_objs['optimizer']

        if not optimizer_wrapper.instance:
            optimizer_wrapper.restore_instance({'params': parameters})

        return optimizer_wrapper.instance


class ResnetTrainWrapper(StateDictRestorableObjectWrapper):

    def restore_instance(self, file_pers_service: AbstractFilePersistenceService,
                         dict_pers_service: AbstractDictPersistenceService, restore_root: str):
        state_dict = {}

        restored_dict = dict_pers_service.recover_dict(self.store_id, RESTORABLE_OBJECT)
        state_objs = restored_dict[STATE_DICT]

        # NOTE: Dataloader instance is loaded in the train routine
        state_dict['optimizer'] = OptimizerWrapper.load(
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


class OptimizerWrapper(StateFileRestorableObjectWrapper):

    def _save_instance_state(self, path):
        if self.instance:
            state_dict = self.instance.state_dict()
            torch.save(state_dict, path)

    def _restore_instance_state(self, path):
        self.instance.load_state_dict(torch.load(path))

import torch

from mmlib.deterministic import set_deterministic
from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from schema.restorable_object import TrainService, OptimizerWrapper, StateDictRestorableObjectWrapper, \
    RESTORABLE_OBJECT, STATE_DICT, RestorableObjectWrapper
from util.init_from_file import create_object


class ImagenetTrainService(TrainService):
    def train(self, model: torch.nn.Module, number_batches=None):

        set_deterministic()

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


class ImagenetTrainWrapper(StateDictRestorableObjectWrapper):

    def load_all_fields(self, file_pers_service: AbstractFilePersistenceService,
                        dict_pers_service: AbstractDictPersistenceService, restore_root: str, load_ref_fields=True):
        pass

    def restore_instance(self, file_pers_service: AbstractFilePersistenceService,
                         dict_pers_service: AbstractDictPersistenceService, restore_root: str):
        state_dict = {}

        restored_dict = dict_pers_service.recover_dict(self.store_id, RESTORABLE_OBJECT)
        state_objs = restored_dict[STATE_DICT]

        print('---------------###########--------------')
        # NOTE: Dataloader instance is loaded in the train routine
        state_dict['optimizer'] = OptimizerWrapper.load(
            state_objs['optimizer'], file_pers_service, dict_pers_service, restore_root)
        print('optimizer: {}'.format(state_dict['optimizer'].store_id))

        data_wrapper = RestorableObjectWrapper.load(
            state_objs['data'], file_pers_service, dict_pers_service, restore_root)
        state_dict['data'] = data_wrapper
        data_wrapper.restore_instance()
        print('data: {}'.format(state_dict['data'].store_id))

        dataloader = RestorableObjectWrapper.load(
            state_objs['dataloader'], file_pers_service, dict_pers_service, restore_root)
        state_dict['dataloader'] = dataloader
        dataloader.restore_instance(ref_type_args={'dataset': data_wrapper.instance})
        print('dataloader: {}'.format(state_dict['dataloader'].store_id))

        print('---------------------------------------')

        self.instance = create_object(code=self.code, class_name=self.class_name)
        self.instance.state_objs = state_dict

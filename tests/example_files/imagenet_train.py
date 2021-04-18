import torch

from mmlib.deterministic import set_deterministic
from mmlib.persistence import FilePersistenceService, DictPersistenceService
from schema.restorable_object import TrainService, OptimizerWrapper, StateDictRestorableObjectWrapper, \
    RESTORABLE_OBJECT, STATE_DICT, RestorableObjectWrapper
from util.init_from_file import create_object

DATA = 'data'
DATALOADER = 'dataloader'
OPTIMIZER = 'optimizer'


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
        dataloader = self.state_objs[DATALOADER].instance
        return dataloader

    def _get_optimizer(self, parameters):
        optimizer_wrapper: OptimizerWrapper = self.state_objs[OPTIMIZER]

        if not optimizer_wrapper.instance:
            optimizer_wrapper.restore_instance({'params': parameters})

        return optimizer_wrapper.instance


class ImagenetTrainWrapper(StateDictRestorableObjectWrapper):

    def restore_instance(self, file_pers_service: FilePersistenceService,
                         dict_pers_service: DictPersistenceService, restore_root: str):
        state_dict = {}

        restored_dict = dict_pers_service.recover_dict(self.store_id, RESTORABLE_OBJECT)
        state_objs = restored_dict[STATE_DICT]

        state_dict[OPTIMIZER] = OptimizerWrapper.load(
            state_objs[OPTIMIZER], file_pers_service, dict_pers_service, restore_root, True, True)

        data_wrapper = RestorableObjectWrapper.load(
            state_objs[DATA], file_pers_service, dict_pers_service, restore_root, True, True)
        state_dict[DATA] = data_wrapper
        data_wrapper.restore_instance()

        # NOTE: Dataloader instance is loaded in the train routine
        dataloader = RestorableObjectWrapper.load(
            state_objs[DATALOADER], file_pers_service, dict_pers_service, restore_root, True, True)
        state_dict[DATALOADER] = dataloader
        dataloader.restore_instance(ref_type_args={'dataset': data_wrapper.instance})

        self.instance = create_object(code=self.code.path, class_name=self.class_name)
        self.instance.state_objs = state_dict

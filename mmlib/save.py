import abc
import os
import tempfile

import torch

from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from mmlib.save_info import ModelSaveInfo
from schema.dataset import Dataset
from schema.inference_info import InferenceInfo
from schema.model_info import ModelInfo, MODEL_INFO
from schema.recover_info import FullModelRecoverInfo, ProvenanceRecoverInfo
from schema.restorable_object import RestoredModelInfo
from schema.store_type import ModelStoreType
from schema.train_info import TrainInfo
from util.init_from_file import create_object, create_type

RESTORE_PATH = 'restore_path'

MODEL_WEIGHTS = 'model_weights'


# Future work, se if it would make sense to use protocol here
class AbstractSaveService(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def save_model(self, model_save_info: ModelSaveInfo) -> str:
        """
        Saves a model together with the given metadata.
        :param model_save_info: An instance of ModelSaveInfo providing all the info needed to save the model.
        :return: Returns the id that was used to store the model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def recover_model(self, model_id: str) -> RestoredModelInfo:
        """
        Recovers a the model and metadata identified by the given model id.
        :param model_id: The id to identify the model with.
        :return: The recovered model and metadata bundled in an object of type ModelRestoreInfo.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def model_save_size(self, model_id: str) -> int:
        """
        Calculates and returns the amount of bytes that are used for storing the model.
        :param model_id: The id to identify the model.
        :return: The amount of bytes used to store the model.
        """
        raise NotImplementedError

    def all_model_ids(self) -> [str]:
        """
        Retuns a list of all stored model_ids
        """
        raise NotImplementedError


class BaselineSaveService(AbstractSaveService):
    """A Service that offers functionality to store PyTorch models by making use of a persistence service.
         The metadata is stored in JSON like dictionaries, files and weights are stored as files."""

    def __init__(self, file_pers_service: AbstractFilePersistenceService,
                 dict_pers_service: AbstractDictPersistenceService):
        """
        :param file_pers_service: An instance of AbstractFilePersistenceService that is used to store files.
        :param dict_pers_service: An instance of AbstractDictPersistenceService that is used to store metadata as dicts.
        """
        self._file_pers_service = file_pers_service
        self._dict_pers_service = dict_pers_service

    def save_model(self, model_save_info: ModelSaveInfo) -> str:
        self._check_consistency(model_save_info)

        # usually we would consider at this bit how we best store the given model
        # but since this is the baseline service we just store the full model every time.
        model_id = self._save_full_model(model_save_info)

        return model_id

    def recover_model(self, model_id: str) -> RestoredModelInfo:
        # in this baseline approach we always store the full model (pickled weights + code)

        with tempfile.TemporaryDirectory() as tmp_path:
            model_info = ModelInfo.load(model_id, self._file_pers_service, self._dict_pers_service, tmp_path,
                                        load_recursive=True)

            # recover model form info
            recover_info: FullModelRecoverInfo = model_info.recover_info

            model = create_object(recover_info.model_code_file_path, recover_info.model_class_name)
            s_dict = self._recover_pickled_weights(recover_info.weights_file_path)
            model.load_state_dict(s_dict)

            restored_model_info = RestoredModelInfo(model=model, inference_info=model_info.inference_info)

        return restored_model_info

    def model_save_size(self, model_id: str) -> int:
        with tempfile.TemporaryDirectory() as tmp_path:
            model_info = ModelInfo.load(model_id, self._file_pers_service, self._dict_pers_service, tmp_path)

        return model_info.size_in_bytes(self._file_pers_service, self._dict_pers_service)

    def all_model_ids(self) -> [str]:
        return self._dict_pers_service.all_ids_for_type(MODEL_INFO)

    def _check_consistency(self, model_save_info):
        assert True, 'nothing checked so far'

    def _save_full_model(self, model_save_info: ModelSaveInfo) -> str:

        with tempfile.TemporaryDirectory() as tmp_path:
            weights_path = self._pickle_weights(model_save_info.model, tmp_path)

            derived_from = model_save_info.base_model if model_save_info.base_model else None

            if derived_from and not (model_save_info.code or model_save_info.class_name):
                # create separate dir to avoid naming conflicts
                restore_dir = os.path.join(tmp_path, 'restore')
                os.mkdir(restore_dir)

                model_info = ModelInfo.load(derived_from, self._file_pers_service, self._dict_pers_service, restore_dir)
                recover_info: FullModelRecoverInfo = model_info.recover_info

                model_save_info.code = recover_info.model_code_file_path
                model_save_info.class_name = recover_info.model_class_name

            # if the model to store is not derived from another model code and class name have to me defined
            recover_info = FullModelRecoverInfo(weights_file_path=weights_path,
                                                model_code_file_path=model_save_info.code,
                                                model_class_name=model_save_info.class_name)

            inference_info = None
            if model_save_info.inference_info:
                inference_info = InferenceInfo(data_wrapper=model_save_info.inference_info.data_wrapper,
                                               dataloader=model_save_info.inference_info.dataloader,
                                               pre_processor=model_save_info.inference_info.pre_processor,
                                               environment=model_save_info.inference_info.environment)

            model_info = ModelInfo(store_type=ModelStoreType.PICKLED_WEIGHTS, recover_info=recover_info,
                                   derived_from_id=derived_from, inference_info=inference_info)

            model_info_id = model_info.persist(self._file_pers_service, self._dict_pers_service)

            return model_info_id

    def _pickle_weights(self, model, save_path):
        # store pickle dump of model
        weight_path = os.path.join(save_path, MODEL_WEIGHTS)
        torch.save(model.state_dict(), weight_path)

        return weight_path

    def _recover_pickled_weights(self, weights_file):
        state_dict = torch.load(weights_file)

        return state_dict

    def _get_store_type(self, model_id: str):
        with tempfile.TemporaryDirectory() as tmp_path:
            model_info = ModelInfo.load(model_id, self._file_pers_service, self._dict_pers_service, tmp_path)
            return model_info.store_type

    def _get_base_model(self, model_id: str):
        with tempfile.TemporaryDirectory() as tmp_path:
            model_info = ModelInfo.load(model_id, self._file_pers_service, self._dict_pers_service, tmp_path)
            return model_info.derived_from


class ProvenanceSaveService(BaselineSaveService):

    def __init__(self, file_pers_service: AbstractFilePersistenceService,
                 dict_pers_service: AbstractDictPersistenceService):
        # baseline_save_service: BaselineSaveService):
        """
        :param file_pers_service: An instance of AbstractFilePersistenceService that is used to store files.
        :param dict_pers_service: An instance of AbstractDictPersistenceService that is used to store metadata as dicts.
        # :param baseline_save_service: An instance of BaselineSaveService that is used to store "full models"
        """
        super().__init__(file_pers_service, dict_pers_service)

    def save_model(self, model_save_info: ModelSaveInfo) -> str:
        if model_save_info.base_model is None:
            # if the base model is none, then we have to store the model as a full model
            return super().save_model(model_save_info)
        else:
            self._check_consistency(model_save_info)

            model_id = self._save_provenance_model(model_save_info)

            return model_id

    def recover_model(self, model_id: str) -> RestoredModelInfo:

        base_model_id = self._get_base_model(model_id)
        if base_model_id is None:
            # if there is no base model the current model's store type must be PickledWeights
            store_type = self._get_store_type(model_id)
            assert store_type == ModelStoreType.PICKLED_WEIGHTS, \
                'for all other model types then ModelStoreType.PICKLED_WEIGHTS we need a base model'
            return super().recover_model(model_id)
        else:
            # if there is a base model we first have to restore the base model to continue training base on it
            base_model_store_type = self._get_store_type(base_model_id)
            base_model_info = self._recover_base_model(base_model_id, base_model_store_type)
            base_model = base_model_info.model

            with tempfile.TemporaryDirectory() as tmp_path:
                restore_dir = os.path.join(tmp_path, RESTORE_PATH)
                os.mkdir(restore_dir)

                model_info = ModelInfo.load(model_id, self._file_pers_service, self._dict_pers_service, restore_dir,
                                            load_recursive=True)
                recover_info: ProvenanceRecoverInfo = model_info.recover_info

                train_service = recover_info.train_info.train_service_wrapper.instance
                train_kwargs = recover_info.train_info.train_kwargs
                train_service.train(base_model, **train_kwargs)

                restored_model_info = RestoredModelInfo(model=base_model)

                # because we trained it here the base_model is the updated version
                return restored_model_info

    def model_save_size(self, model_id: str) -> int:
        pass

    def _check_consistency(self, model_save_info):
        # TODO
        pass

    def _save_provenance_model(self, model_save_info):
        tw_class_name = model_save_info.prov_rec_info.train_info.train_wrapper_class_name
        tw_code = model_save_info.prov_rec_info.train_info.train_wrapper_code
        type_ = create_type(code=tw_code, type_name=tw_class_name)
        train_service_wrapper = type_(
            class_name=model_save_info.prov_rec_info.train_info.train_service_class_name,
            code=model_save_info.prov_rec_info.train_info.train_service_code,
            instance=model_save_info.prov_rec_info.train_info.train_service
        )
        dataset = Dataset(model_save_info.prov_rec_info.raw_dataset)
        train_info = TrainInfo(
            ts_wrapper=train_service_wrapper,
            ts_wrapper_code=tw_code,
            ts_wrapper_class_name=tw_class_name,
            train_kwargs=model_save_info.prov_rec_info.train_info.train_kwargs,
            environment=model_save_info.prov_rec_info.train_info.environment
        )

        prov_recover_info = ProvenanceRecoverInfo(
            dataset=dataset,
            model_code_file_path=model_save_info.prov_rec_info.model_code,
            model_class_name=model_save_info.prov_rec_info.model_class_name,
            train_info=train_info
        )

        derived_from = model_save_info.base_model if model_save_info.base_model else None

        # TODO implement inference info
        inference_info = None

        model_info = ModelInfo(store_type=ModelStoreType.PROVENANCE, recover_info=prov_recover_info,
                               derived_from_id=derived_from, inference_info=inference_info)

        model_info_id = model_info.persist(self._file_pers_service, self._dict_pers_service)

        return model_info_id

    def _recover_base_model(self, base_model_id, base_model_store_type):
        if base_model_store_type == ModelStoreType.PICKLED_WEIGHTS:
            return super().recover_model(model_id=base_model_id)
        elif base_model_store_type == ModelStoreType.PROVENANCE:
            return self.recover_model(model_id=base_model_id)
        else:
            raise NotImplementedError

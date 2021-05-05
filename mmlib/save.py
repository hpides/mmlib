import abc
import os
import tempfile
import warnings

import torch

from mmlib.equal import tensor_equal
from mmlib.persistence import FilePersistenceService, DictPersistenceService
from mmlib.recover_validation import RecoverValidationService
from mmlib.save_info import ModelSaveInfo, ProvModelSaveInfo
from mmlib.track_env import compare_env_to_current
from schema.dataset import Dataset
from schema.file_reference import FileReference
from schema.model_info import ModelInfo, MODEL_INFO
from schema.recover_info import FullModelRecoverInfo, ProvenanceRecoverInfo, WeightsUpdateRecoverInfo
from schema.restorable_object import RestoredModelInfo
from schema.store_type import ModelStoreType
from schema.train_info import TrainInfo
from util.init_from_file import create_object, create_type
from util.weight_dict_merkle_tree import WeightDictMerkleTree

PICKLED_MODEL_WEIGHTS = 'pickled_model_weights'

WEIGHTS_PATCH = "weights_patch"

RESTORE_PATH = 'restore_path'

MODEL_WEIGHTS = 'model_weights'


# Future work, se if it would make sense to use protocol here
class AbstractSaveService(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def save_model(self, model_save_info: ModelSaveInfo) -> str:
        """
        Saves a model together with the given metadata.
        :param model_save_info: An instance of ModelSaveInfo providing all the info needed to save the model.
         process. - If set, time consumption for save process might rise.
        :return: Returns the id that was used to store the model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def recover_model(self, model_id: str, execute_checks: bool = False,
                      recover_val_service: RecoverValidationService = None) -> RestoredModelInfo:
        """
        Recovers a the model and metadata identified by the given model id.
        :param model_id: The id to identify the model with.
        :param execute_checks: Indicates if additional checks should be performed to ensure a correct recovery of
        the model. If set to True setting it to True recover_val_service must be given - might decrease the performance.
        :param recover_val_service: An instance of RecoverValidationService.
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

    @abc.abstractmethod
    def all_model_ids(self) -> [str]:
        """
        Retuns a list of all stored model_ids
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save_validation_info(self, model: torch.nn.Module, model_id: str, dummy_input_shape: [int],
                             recover_val_service: RecoverValidationService):
        """
        Saves validation info for the given model. This info can be used to validate the model when it is restored.
        To activate the check when restoring a model set the parameter execute_checks = True
        :param model: The model to save the validation info for.
        :param model_id: The id of the model to save the recover information for.
        :param dummy_input_shape: The input shape to generate a dummy output for the model.
        :param recover_val_service: An instance of RecoverValidationService.
        fields are 'model' and 'dummy_input_shape'.
        """
        raise NotImplementedError


class BaselineSaveService(AbstractSaveService):
    """A Service that offers functionality to store PyTorch models by making use of a persistence service.
         The metadata is stored in JSON like dictionaries, files and weights are stored as files."""

    def __init__(self, file_pers_service: FilePersistenceService,
                 dict_pers_service: DictPersistenceService):
        """
        :param file_pers_service: An instance of FilePersistenceService that is used to store files.
        :param dict_pers_service: An instance of DictPersistenceService that is used to store metadata as dicts.
        """
        self._file_pers_service = file_pers_service
        self._dict_pers_service = dict_pers_service

    def save_model(self, model_save_info: ModelSaveInfo) -> str:
        self._check_consistency(model_save_info)

        # usually we would consider at this bit how we best store the given model
        # but since this is the baseline service we just store the full model every time.
        model_id = self._save_full_model(model_save_info)

        return model_id

    def recover_model(self, model_id: str, execute_checks: bool = False,
                      recover_val_service: RecoverValidationService = None) -> RestoredModelInfo:
        # in this baseline approach we always store the full model (pickled weights + code)

        with tempfile.TemporaryDirectory() as tmp_path:
            model_info = ModelInfo.load(model_id, self._file_pers_service, self._dict_pers_service, tmp_path,
                                        load_recursive=True, load_files=True)

            # recover model form info
            recover_info: FullModelRecoverInfo = model_info.recover_info

            model = create_object(recover_info.model_code.path, recover_info.model_class_name)
            s_dict = self._recover_pickled_weights(recover_info.weights_file.path)
            model.load_state_dict(s_dict)

            restored_model_info = RestoredModelInfo(model=model)

            if execute_checks:
                self._execute_checks(model, model_info, recover_val_service)

        return restored_model_info

    def model_save_size(self, model_id: str) -> int:
        with tempfile.TemporaryDirectory() as tmp_path:
            model_info = ModelInfo.load(model_id, self._file_pers_service, self._dict_pers_service, tmp_path)

        return model_info.size_in_bytes(self._file_pers_service, self._dict_pers_service)

    def all_model_ids(self) -> [str]:
        return self._dict_pers_service.all_ids_for_type(MODEL_INFO)

    def save_validation_info(self, model, model_id, dummy_input_shape, recover_val_service):
        recover_val_service.save_recover_val_info(model, model_id, dummy_input_shape)

    def _check_consistency(self, model_save_info):
        # when storing a full model we need the following information
        # the model itself
        assert model_save_info.model, 'model is not set'
        # the model code
        assert model_save_info.model_code, 'model code is not set'
        # the class name of the model
        assert model_save_info.model_class_name, 'model class name is not set'

    def _save_full_model(self, model_save_info: ModelSaveInfo, add_weights_hash_info=True) -> str:

        with tempfile.TemporaryDirectory() as tmp_path:
            weights_path = self._pickle_weights(model_save_info.model, tmp_path)

            base_model = model_save_info.base_model if model_save_info.base_model else None

            # models are recovered in a tmp directory and only the model object is returned
            # this is why the inferred model code path might not exists anymore, we have to check this
            # and if it is not existing anymore, we have to restore teh code for the base model

            if not os.path.isfile(model_save_info.model_code):
                assert base_model, 'code not given and no base model'
                base_model_info = ModelInfo.load(base_model, self._file_pers_service, self._dict_pers_service, tmp_path)
                model_code = self._restore_code_from_base_model(base_model_info, tmp_path)
                model_save_info.model_code = model_code.path

            recover_info = FullModelRecoverInfo(weights_file=FileReference(path=weights_path),
                                                model_code=FileReference(path=model_save_info.model_code),
                                                model_class_name=model_save_info.model_class_name)

            weights_hash_info = _get_weights_info_hash(add_weights_hash_info, model_save_info)

            model_info = ModelInfo(store_type=ModelStoreType.FULL_MODEL, recover_info=recover_info,
                                   derived_from_id=base_model, weights_hash_info=weights_hash_info)

            model_info_id = model_info.persist(self._file_pers_service, self._dict_pers_service)

            return model_info_id

    def _restore_code_from_base_model(self, model_info: ModelInfo, tmp_path):
        assert isinstance(model_info, ModelInfo)

        code, _ = self._restore_code_and_class_name(model_info, tmp_path)
        return code

    def _find_nearest_full_model_info(self, model_info, restore_dir):
        current_model_info = model_info
        while not (hasattr(current_model_info, 'store_type') and
                   current_model_info.store_type == ModelStoreType.FULL_MODEL):
            base_model_id = current_model_info.derived_from
            base_model_info = ModelInfo.load(
                obj_id=base_model_id,
                file_pers_service=self._file_pers_service,
                dict_pers_service=self._dict_pers_service,
                restore_root=restore_dir,
            )
            current_model_info = base_model_info
        full_model_info: ModelInfo = current_model_info
        return full_model_info

    def _restore_code_and_class_name(self, model_info: ModelInfo, tmp_path):
        full_model_info = self._find_nearest_full_model_info(model_info, tmp_path)
        assert isinstance(full_model_info.recover_info, FullModelRecoverInfo), 'model info has to be full model info'
        recover_info: FullModelRecoverInfo = full_model_info.recover_info
        # make sure all required fields are loaded
        if not (recover_info.model_class_name and recover_info.model_code):
            recover_info.load_all_fields(self._file_pers_service, self._dict_pers_service, tmp_path,
                                         load_recursive=True, load_files=False)
        class_name = recover_info.model_class_name
        code: FileReference = recover_info.model_code
        self._file_pers_service.recover_file(code, tmp_path)

        return code, class_name

    def _pickle_weights(self, model, save_path):
        # store pickle dump of model weights
        state_dict = model.state_dict()
        weight_path = self._pickle_state_dict(state_dict, save_path)

        return weight_path

    def _pickle_state_dict(self, state_dict, save_path):
        weight_path = os.path.join(save_path, MODEL_WEIGHTS)
        torch.save(state_dict, weight_path)
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

    def _execute_checks(self, model: torch.nn.Module, model_info: ModelInfo,
                        recover_val_service: RecoverValidationService):
        assert recover_val_service, 'if execute_checks is True recover_val_service must be given'
        model_id = model_info.store_id
        try:
            valid_recovery = recover_val_service.check_recover_val(model_id, model)
            assert valid_recovery, 'The current given model differs from the model that was stored'
        except IndexError:
            warnings.warn('no recover validation info found'
                          ' - check that save_validation_info=True when saving model')


class WeightUpdateSaveService(BaselineSaveService):

    def __init__(self, file_pers_service: FilePersistenceService,
                 dict_pers_service: DictPersistenceService):
        """
        :param file_pers_service: An instance of FilePersistenceService that is used to store files.
        :param dict_pers_service: An instance of DictPersistenceService that is used to store metadata as dicts.
        """
        super().__init__(file_pers_service, dict_pers_service)

    def save_model(self, model_save_info: ModelSaveInfo) -> str:

        # as a first step we have to find out if we have to store a full model first or if we can store only the update
        # if there is no base model given, we can not compute any updates -> we have to sore the full model
        if not self._base_model_given(model_save_info):
            return super().save_model(model_save_info)
        else:
            # if there is a base model, we can store the update and for a restore refer to the base model
            return self._save_updated_model(model_save_info)

    def recover_model(self, model_id: str, execute_checks: bool = False,
                      recover_val_service: RecoverValidationService = None) -> RestoredModelInfo:

        store_type = self._get_store_type(model_id)

        if store_type == ModelStoreType.FULL_MODEL:
            return super().recover_model(model_id)
        else:
            return self._recover_from_weight_update(model_id, execute_checks, recover_val_service)

    def _recover_from_weight_update(self, model_id, execute_checks, recover_val_service):
        with tempfile.TemporaryDirectory() as tmp_path:
            model_info = ModelInfo.load(model_id, self._file_pers_service, self._dict_pers_service, tmp_path,
                                        load_recursive=True, load_files=True)

            recover_info: WeightsUpdateRecoverInfo = model_info.recover_info

            if recover_info.independent:
                recovered_model = self._restore_independent_update(model_info, tmp_path)
            else:
                recovered_model = self._restore_dependent_update(model_info)

            restored_model_info = RestoredModelInfo(model=recovered_model)

            if execute_checks:
                self._execute_checks(recovered_model, model_info, recover_val_service)

        return restored_model_info

    def _restore_independent_update(self, model_info, tmp_path):
        model_code, model_class_name = self._restore_code_and_class_name(model_info, tmp_path)
        recover_info: WeightsUpdateRecoverInfo = model_info.recover_info

        model = create_object(model_code.path, model_class_name)
        if recover_info.update_type:  # here we should actually check the type
            s_dict = self._recover_pickled_weights(recover_info.update.path)
            model.load_state_dict(s_dict)

        return model

    def _restore_dependent_update(self, model_info):
        recover_info: WeightsUpdateRecoverInfo = model_info.recover_info
        base_model_info = self.recover_model(model_info.derived_from)
        base_model = base_model_info.model
        weights_patch = torch.load(recover_info.update.path)
        self._apply_weight_patch(base_model, weights_patch)

        return base_model

    def _save_updated_model(self, model_save_info, add_weights_hash_info=True):
        base_model_id = model_save_info.base_model
        assert base_model_id, 'no base model given'

        with tempfile.TemporaryDirectory() as tmp_path:
            weights_hash_info = _get_weights_info_hash(add_weights_hash_info, model_save_info)

            weights_update, update_type, independent = \
                self._generate_weights_update(model_save_info, base_model_id, weights_hash_info, tmp_path)

            recover_info = WeightsUpdateRecoverInfo(update=FileReference(path=weights_update), update_type=update_type,
                                                    independent=independent)

            model_info = ModelInfo(store_type=ModelStoreType.WEIGHT_UPDATES, recover_info=recover_info,
                                   derived_from_id=base_model_id, weights_hash_info=weights_hash_info)

            model_info_id = model_info.persist(self._file_pers_service, self._dict_pers_service)

            return model_info_id

    def _base_model_given(self, model_save_info):
        return model_save_info.base_model is not None

    def _generate_weights_update(self, model_save_info, base_model_id, weights_hash_info, tmp_path):
        base_model_info = ModelInfo.load(base_model_id, self._file_pers_service, self._dict_pers_service, tmp_path)
        current_model_weights = model_save_info.model.state_dict()

        if base_model_info.weights_hash_info:
            diff_indices = weights_hash_info.get_diff_indices(base_model_info.weights_hash_info)
            # all keys that do to differ can be deleted -> we only need the diff of the weights
            del_keys = []
            for i, key in enumerate(current_model_weights.keys()):
                if i not in diff_indices:
                    del_keys.append(key)

            # delete all weights that have not changed compared to the base model
            for key in del_keys:
                del current_model_weights[key]

            return current_model_weights, WEIGHTS_PATCH, False
        else:
            # if there is no weights hash info given we have to fall back and load the base models
            base_model_info = self.recover_model(base_model_id)
            base_model_weights = base_model_info.model.state_dict()
            current_model_weights = model_save_info.model.state_dict()

            weights_patch = self._state_dict_patch(base_model_weights, current_model_weights)
            if len(weights_patch.keys()) < len(base_model_weights.keys()):
                # if the patch actually saves something
                model_weights = super()._pickle_state_dict(weights_patch, tmp_path)
                return model_weights, WEIGHTS_PATCH, False
            else:
                model_weights = self._pickle_weights(current_model_weights, tmp_path)
                return model_weights, PICKLED_MODEL_WEIGHTS, True

    def _state_dict_patch(self, base_model_weights, current_model_weights):
        assert base_model_weights.keys() == current_model_weights.keys(), 'given state dicts are not compatible'
        for k in list(current_model_weights.keys()):
            if tensor_equal(base_model_weights[k], current_model_weights[k]):
                del current_model_weights[k]

        return current_model_weights

    def _apply_weight_patch(self, base_model: torch.nn.Module, weights_patch):
        patched_state_dict = base_model.state_dict()
        for k, patch in weights_patch.items():
            patched_state_dict[k] = patch

        return base_model.load_state_dict(patched_state_dict)


class ProvenanceSaveService(BaselineSaveService):

    def __init__(self, file_pers_service: FilePersistenceService,
                 dict_pers_service: DictPersistenceService):
        """
        :param file_pers_service: An instance of FilePersistenceService that is used to store files.
        :param dict_pers_service: An instance of DictPersistenceService that is used to store metadata as dicts.
        """
        super().__init__(file_pers_service, dict_pers_service)

    def save_model(self, model_save_info: ModelSaveInfo) -> str:
        if model_save_info.base_model is None or not isinstance(model_save_info, ProvModelSaveInfo):
            # if the base model is none or model save info does not provide provenance save info we have to store the
            # model as a full model
            return super().save_model(model_save_info)
        else:
            return self._save_provenance_model(model_save_info)

    def recover_model(self, model_id: str, execute_checks: bool = False,
                      recover_val_service: RecoverValidationService = None) -> RestoredModelInfo:

        base_model_id = self._get_base_model(model_id)
        if base_model_id is None:
            # if there is no base model the current model's store type must be FULL_MODEL
            store_type = self._get_store_type(model_id)
            assert store_type == ModelStoreType.FULL_MODEL, \
                'for all other model types then ModelStoreType.PICKLED_WEIGHTS we need a base model'
            return super().recover_model(model_id, execute_checks)
        else:
            # if there is a base model we first have to restore the base model to continue training base on it
            base_model_store_type = self._get_store_type(base_model_id)
            base_model_info = self._recover_base_model(base_model_id, base_model_store_type)
            base_model = base_model_info.model

            with tempfile.TemporaryDirectory() as tmp_path:
                # TODO maybe can be replaced when using FileRef Object
                restore_dir = os.path.join(tmp_path, RESTORE_PATH)
                os.mkdir(restore_dir)

                model_info = ModelInfo.load(model_id, self._file_pers_service, self._dict_pers_service, restore_dir,
                                            load_recursive=True, load_files=True)
                recover_info: ProvenanceRecoverInfo = model_info.recover_info

                train_service = recover_info.train_info.train_service_wrapper.instance
                train_kwargs = recover_info.train_info.train_kwargs
                train_service.train(base_model, **train_kwargs)

                # because we trained it here the base_model is the updated version
                restored_model = base_model
                restored_model_info = RestoredModelInfo(model=restored_model)

                if execute_checks:
                    self._execute_checks(restored_model, model_info, recover_val_service)

                return restored_model_info

    def model_save_size(self, model_id: str) -> int:
        pass

    def _save_provenance_model(self, model_save_info):
        model_info = self._build_prov_model_info(model_save_info)

        model_info_id = model_info.persist(self._file_pers_service, self._dict_pers_service)

        return model_info_id

    def _build_prov_model_info(self, model_save_info):
        tw_class_name = model_save_info.train_info.train_wrapper_class_name
        tw_code = FileReference(path=model_save_info.train_info.train_wrapper_code)
        type_ = create_type(code=tw_code.path, type_name=tw_class_name)
        train_service_wrapper = type_(
            instance=model_save_info.train_info.train_service
        )
        dataset = Dataset(FileReference(path=model_save_info.raw_dataset))
        train_info = TrainInfo(
            ts_wrapper=train_service_wrapper,
            ts_wrapper_code=tw_code,
            ts_wrapper_class_name=tw_class_name,
            train_kwargs=model_save_info.train_info.train_kwargs,
        )
        prov_recover_info = ProvenanceRecoverInfo(
            dataset=dataset,
            train_info=train_info,
            environment=model_save_info.train_info.environment
        )
        derived_from = model_save_info.base_model if model_save_info.base_model else None
        model_info = ModelInfo(store_type=ModelStoreType.PROVENANCE, recover_info=prov_recover_info,
                               derived_from_id=derived_from)
        return model_info

    def _recover_base_model(self, base_model_id, base_model_store_type):
        if base_model_store_type == ModelStoreType.FULL_MODEL:
            return super().recover_model(model_id=base_model_id)
        elif base_model_store_type == ModelStoreType.PROVENANCE:
            return self.recover_model(model_id=base_model_id)
        else:
            raise NotImplementedError

    def _execute_checks(self, model: torch.nn.Module, model_info: ModelInfo,
                        recover_val_service: RecoverValidationService):
        super()._execute_checks(model, model_info, recover_val_service)

        # check environment
        recover_info: ProvenanceRecoverInfo = model_info.recover_info
        envs_match = compare_env_to_current(recover_info.environment)
        assert envs_match, 'The current environment and the environment that was used to '


def _get_weights_info_hash(add_weights_hash_info, model_save_info):
    weights_hash_info = None
    if add_weights_hash_info:
        assert model_save_info.model, "to compute a weights info hash the a model has to be given"
        weights_hash_info = WeightDictMerkleTree(model_save_info.model.state_dict())
    return weights_hash_info

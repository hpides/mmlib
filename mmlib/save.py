import abc
import os
import tempfile
import warnings

import torch

from mmlib.equal import tensor_equal
from mmlib.persistence import FilePersistenceService, DictPersistenceService
from mmlib.save_info import ModelSaveInfo, ProvModelSaveInfo
from mmlib.track_env import compare_env_to_current
from mmlib.schema import Dataset
from mmlib.schema import FileReference
from mmlib.schema import ModelInfo, MODEL_INFO
from mmlib.schema import FullModelRecoverInfo, ProvenanceRecoverInfo, WeightsUpdateRecoverInfo
from mmlib.schema.restorable_object import RestoredModelInfo
from mmlib.schema import ModelStoreType
from mmlib.schema.train_info import TrainInfo
from mmlib.util.helper import log_start, log_stop
from mmlib.util.init_from_file import create_object, create_type
from mmlib.util.weight_dict_merkle_tree import WeightDictMerkleTree, THIS, OTHER

PROVENANCE = 'provenance'

PARAM_UPDATE = 'param_update'

BASELINE = 'baseline'

START = 'START'
STOP = 'STOP'

PICKLED_MODEL_PARAMETERS = 'pickled_model_parameters'

PARAMETERS_PATCH = "parameters_patch"

RESTORE_PATH = 'restore_path'

MODEL_WEIGHTS = 'model_weights.pt'


# Future work, se if it would make sense to use protocol here
class AbstractSaveService(metaclass=abc.ABCMeta):

    def __init__(self, logging=False):
        """
        :param logging: Flag that indicates if logging is turned in for this service.
        """
        self.logging = logging

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
    def recover_model(self, model_id: str, execute_checks: bool = True) -> RestoredModelInfo:
        """
        Recovers a the model and metadata identified by the given model id.
        :param model_id: The id to identify the model with.
        :param execute_checks: Indicates if additional checks should be performed to ensure a correct recovery of
        the model.
        :return: The recovered model and metadata bundled in an object of type ModelRestoreInfo.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def model_save_size(self, model_id: str) -> dict:
        """
        Gives detailed information about the storage consumption of a model.
        :param model_id: The id to identify the model.
        :return: Detailed information about the storage consumption of a model -- size in bytes.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def all_model_ids(self) -> [str]:
        """
        Retuns a list of all stored model_ids
        """
        raise NotImplementedError


class BaselineSaveService(AbstractSaveService):
    """A Service that offers functionality to store PyTorch models by making use of a persistence service.
         The metadata is stored in JSON like dictionaries, files and weights are stored as files."""

    def __init__(self, file_pers_service: FilePersistenceService, dict_pers_service: DictPersistenceService,
                 logging=False):
        """
        :param file_pers_service: An instance of FilePersistenceService that is used to store files.
        :param dict_pers_service: An instance of DictPersistenceService that is used to store metadata as dicts.
        :param logging: Flag that indicates if logging is turned in for this service.
        """
        super().__init__(logging)
        self._file_pers_service = file_pers_service
        self._dict_pers_service = dict_pers_service
        self._file_pers_service.logging = logging
        self._dict_pers_service.logging = logging

    def save_model(self, model_save_info: ModelSaveInfo) -> str:
        self._check_consistency(model_save_info)

        # usually we would consider at this bit how we best store the given model
        # but since this is the baseline service we just store the full model every time.
        model_id = self._save_full_model(model_save_info)

        return model_id

    def recover_model(self, model_id: str, execute_checks: bool = True) -> RestoredModelInfo:
        # in this baseline approach we always store the full model (pickled weights + code)
        log_all = log_start(self.logging, BASELINE, 'recover_model-{}'.format(model_id), 'all')
        with tempfile.TemporaryDirectory() as tmp_path:
            log_load = log_start(self.logging, BASELINE, 'recover_model', 'load_model_info_rec_files')
            model_info = ModelInfo.load(model_id, self._file_pers_service, self._dict_pers_service, tmp_path,
                                        load_recursive=True, load_files=True)
            log_stop(self.logging, log_load)

            log_recover = log_start(self.logging, BASELINE, 'recover_model', 'recover_from_info')
            # recover model form info
            recover_info: FullModelRecoverInfo = model_info.recover_info

            model = create_object(recover_info.model_code.path, recover_info.model_class_name)
            s_dict = self._recover_pickled_weights(recover_info.parameters_file.path)
            model.load_state_dict(s_dict)

            restored_model_info = RestoredModelInfo(model=model)

            if execute_checks:
                log_check_weights = log_start(self.logging, BASELINE, 'recover_model', '_check_weights')
                self._check_weights(model, model_info)
                log_stop(self.logging, log_check_weights)
                log_check_env = log_start(self.logging, BASELINE, 'recover_model', '_check_env')
                self._check_env(model_info)
                log_stop(self.logging, log_check_env)

        log_stop(self.logging, log_recover)
        log_stop(self.logging, log_all)
        return restored_model_info

    def model_save_size(self, model_id: str) -> dict:
        place_holder = ModelInfo.load_placeholder(model_id)
        size_dict = place_holder.size_info(self._file_pers_service, self._dict_pers_service)

        return size_dict

    def all_model_ids(self) -> [str]:
        return self._dict_pers_service.all_ids_for_type(MODEL_INFO)

    def _check_consistency(self, model_save_info):
        # when storing a full model we need the following information
        # the model itself
        assert model_save_info.model, 'model is not set'
        # the model code
        assert model_save_info.model_code, 'model code is not set'
        # the class name of the model
        assert model_save_info.model_class_name, 'model class name is not set'

    def _save_full_model(self, model_save_info: ModelSaveInfo, add_weights_hash_info=True) -> str:
        log_all = log_start(self.logging, BASELINE, '_save_full_model', 'all')

        with tempfile.TemporaryDirectory() as tmp_path:
            log_pickle = log_start(self.logging, BASELINE, '_save_full_model', 'pickle_weights')
            weights_path = self._pickle_weights(model_save_info.model, tmp_path)
            log_stop(self.logging, log_pickle)

            base_model = model_save_info.base_model if model_save_info.base_model else None

            # models are recovered in a tmp directory and only the model object is returned
            # this is why the inferred model code path might not exists anymore, we have to check this
            # and if it is not existing anymore, we have to restore the code for the base model

            if not os.path.isfile(model_save_info.model_code):
                assert base_model, 'code not given and no base model'
                base_model_info = ModelInfo.load(base_model, self._file_pers_service, self._dict_pers_service, tmp_path)
                model_code = self._restore_code_from_base_model(base_model_info, tmp_path)
                model_save_info.model_code = model_code.path

            recover_info = FullModelRecoverInfo(parameters_file=FileReference(path=weights_path),
                                                model_code=FileReference(path=model_save_info.model_code),
                                                model_class_name=model_save_info.model_class_name,
                                                environment=model_save_info.environment)

            log_weight_hash = log_start(self.logging, BASELINE, '_save_full_model', '_get_weights_hash_info')
            weights_hash_info = _get_weights_hash_info(add_weights_hash_info, model_save_info)
            log_stop(self.logging, log_weight_hash)

            model_info = ModelInfo(store_type=ModelStoreType.FULL_MODEL, recover_info=recover_info,
                                   derived_from_id=base_model, weights_hash_info=weights_hash_info)

            log_persist = log_start(self.logging, BASELINE, '_save_full_model', 'persist_model_info')
            model_info_id = model_info.persist(self._file_pers_service, self._dict_pers_service)
            log_stop(self.logging, log_persist)

            log_stop(self.logging, log_all)

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

    def _check_weights(self, model, model_info):
        if not model_info.weights_hash_info:
            warnings.warn('no weights_hash_info available for this models')
        restored_merkle_tree: WeightDictMerkleTree = model_info.weights_hash_info
        model_merkle_tree = WeightDictMerkleTree.from_state_dict(model.state_dict())
        # NOTE maybe replace assert by throwing exception
        assert restored_merkle_tree == model_merkle_tree, 'The recovered model differs from the model that was stored'

    def _check_env(self, model_info):
        # check environment
        recover_info = model_info.recover_info
        envs_match = compare_env_to_current(recover_info.environment)
        assert envs_match, \
            'The current environment and the environment that was used to when storing the model differ'


class WeightUpdateSaveService(BaselineSaveService):

    def __init__(self, file_pers_service: FilePersistenceService, dict_pers_service: DictPersistenceService,
                 improved_version=True, logging=False):
        """
        :param file_pers_service: An instance of FilePersistenceService that is used to store files.
        :param dict_pers_service: An instance of DictPersistenceService that is used to store metadata as dicts.
        :param logging: Flag that indicates if logging is turned in for this service.
        """
        super().__init__(file_pers_service, dict_pers_service, logging)
        self.improved_version = improved_version

    def save_model(self, model_save_info: ModelSaveInfo) -> str:

        # as a first step we have to find out if we have to store a full model first or if we can store only the update
        # if there is no base model given, we can not compute any updates -> we have to sore the full model
        log_all = log_start(self.logging, PARAM_UPDATE, 'save_model', 'all')
        if not self._base_model_given(model_save_info):
            model_id = super().save_model(model_save_info)
        else:
            # if there is a base model, we can store the update and for a restore refer to the base model
            model_id = self._save_updated_model(model_save_info)

        log_stop(self.logging, log_all)
        return model_id

    def recover_model(self, model_id: str, execute_checks: bool = True) -> RestoredModelInfo:

        log_all = log_start(self.logging, PARAM_UPDATE, 'recover_model-{}'.format(model_id), 'all')
        store_type = self._get_store_type(model_id)

        if store_type == ModelStoreType.FULL_MODEL:
            model = super().recover_model(model_id)
        else:
            model = self._recover_from_weight_update(model_id, execute_checks)

        log_stop(self.logging, log_all)
        return model

    def _recover_from_weight_update(self, model_id, execute_checks):
        log_update = log_start(self.logging, PARAM_UPDATE, 'recover_model', '_recover_from_weight_update')
        with tempfile.TemporaryDirectory() as tmp_path:
            model_info = ModelInfo.load(model_id, self._file_pers_service, self._dict_pers_service, tmp_path,
                                        load_recursive=True, load_files=True)

            recover_info: WeightsUpdateRecoverInfo = model_info.recover_info

            if recover_info.update_type == PICKLED_MODEL_PARAMETERS:
                recovered_model = self._recover_from_full_weights(model_info, tmp_path)
            elif recover_info.update_type == PARAMETERS_PATCH:
                recovered_model = self._recover_from_parameter_patch(model_info)
            else:
                raise NotImplementedError

            restored_model_info = RestoredModelInfo(model=recovered_model)

            if execute_checks:
                log_check_weights = log_start(
                    self.logging, PARAM_UPDATE, '_recover_from_weight_update', '_check_weights')
                self._check_weights(recovered_model, model_info)
                log_stop(self.logging, log_check_weights)

        log_stop(self.logging, log_update)
        return restored_model_info

    def _recover_from_full_weights(self, model_info, tmp_path):
        log = log_start(self.logging, PARAM_UPDATE, '_recover_from_full_weights', 'all')
        model_code, model_class_name = self._restore_code_and_class_name(model_info, tmp_path)
        recover_info: WeightsUpdateRecoverInfo = model_info.recover_info

        model = create_object(model_code.path, model_class_name)
        s_dict = self._recover_pickled_weights(recover_info.update.path)
        model.load_state_dict(s_dict)
        log_stop(self.logging, log)

        return model

    def _recover_from_parameter_patch(self, model_info):
        log = log_start(self.logging, PARAM_UPDATE, '_recover_from_parameter_patch', 'all')
        recover_info: WeightsUpdateRecoverInfo = model_info.recover_info
        base_model_info = self.recover_model(model_info.derived_from)
        base_model = base_model_info.model
        weights_patch = torch.load(recover_info.update.path)
        self._apply_weight_patch(base_model, weights_patch)
        log_stop(self.logging, log)

        return base_model

    def _save_updated_model(self, model_save_info, add_weights_hash_info=True):
        log_all = log_start(self.logging, PARAM_UPDATE, '_save_updated_model', 'all')

        base_model_id = model_save_info.base_model
        assert base_model_id, 'no base model given'

        with tempfile.TemporaryDirectory() as tmp_path:
            log_weights_hash = log_start(self.logging, PARAM_UPDATE, '_save_updated_model', 'get_weights_hash_info')
            weights_hash_info = _get_weights_hash_info(add_weights_hash_info, model_save_info)
            log_stop(self.logging, log_weights_hash)

            log_gen_update = log_start(self.logging, PARAM_UPDATE, '_save_updated_model', 'generate_weights_update')
            weights_update, update_type = \
                self._generate_weights_update(model_save_info, base_model_id, weights_hash_info, tmp_path)
            log_stop(self.logging, log_gen_update)

            recover_info = WeightsUpdateRecoverInfo(update=FileReference(path=weights_update), update_type=update_type)

            model_info = ModelInfo(store_type=ModelStoreType.WEIGHT_UPDATES, recover_info=recover_info,
                                   derived_from_id=base_model_id, weights_hash_info=weights_hash_info)

            log_persist = log_start(self.logging, PARAM_UPDATE, '_save_updated_model', 'persist')
            model_info_id = model_info.persist(self._file_pers_service, self._dict_pers_service)
            log_stop(self.logging, log_persist)

            log_stop(self.logging, log_all)
            return model_info_id

    def _base_model_given(self, model_save_info):
        return model_save_info.base_model is not None

    def _generate_weights_update(self, model_save_info, base_model_id, weights_hash_info, tmp_path):
        base_model_info = ModelInfo.load(base_model_id, self._file_pers_service, self._dict_pers_service, tmp_path)
        current_model_weights = model_save_info.model.state_dict()

        if self.improved_version and base_model_info.weights_hash_info:
            diff_weights, diff_nodes = base_model_info.weights_hash_info.diff(weights_hash_info)
            assert len(diff_nodes[THIS]) == 0 and len(diff_nodes[OTHER]) == 0, \
                'models with different architecture not supported for now'

            weights_patch = current_model_weights.copy()
            # delete all keys that are the same, meaning not in the diff list
            for key in current_model_weights.keys():
                if key not in diff_weights:
                    del weights_patch[key]

            model_weights = super()._pickle_state_dict(weights_patch, tmp_path)
            return model_weights, PARAMETERS_PATCH
        else:
            print('recover base models')
            # if there is no weights hash info given we have to fall back and load the base models
            base_model_info = self.recover_model(base_model_id)
            base_model_weights = base_model_info.model.state_dict()
            current_model_weights = model_save_info.model.state_dict()

            weights_patch = self._state_dict_patch(base_model_weights, current_model_weights)
            if len(weights_patch.keys()) < len(base_model_weights.keys()):
                # if the patch actually saves something
                model_weights = super()._pickle_state_dict(weights_patch, tmp_path)
                return model_weights, PARAMETERS_PATCH
            else:
                model_weights = self._pickle_weights(model_save_info.model, tmp_path)
                return model_weights, PICKLED_MODEL_PARAMETERS

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

    def _execute_checks(self, model: torch.nn.Module, model_info: ModelInfo):
        self._check_weights(model, model_info)


class ProvenanceSaveService(BaselineSaveService):

    def __init__(self, file_pers_service: FilePersistenceService, dict_pers_service: DictPersistenceService,
                 logging=False):
        """
        :param file_pers_service: An instance of FilePersistenceService that is used to store files.
        :param dict_pers_service: An instance of DictPersistenceService that is used to store metadata as dicts.
        :param logging: Flag that indicates if logging is turned in for this service.
        """
        super().__init__(file_pers_service, dict_pers_service, logging)

    def save_model(self, model_save_info: ModelSaveInfo) -> str:
        log_all = log_start(self.logging, PROVENANCE, '_save_model', 'all')
        if model_save_info.base_model is None or not isinstance(model_save_info, ProvModelSaveInfo):
            # if the base model is none or model save info does not provide provenance save info we have to store the
            # model as a full model
            model_id = super().save_model(model_save_info)
        else:
            model_id = self._save_provenance_model(model_save_info)

        log_stop(self.logging, log_all)
        return model_id

    def recover_model(self, model_id: str, execute_checks: bool = True) -> RestoredModelInfo:
        log_all = log_start(self.logging, PROVENANCE, 'recover_model-{}'.format(model_id), 'all')

        base_model_id = self._get_base_model(model_id)
        if self._get_store_type(model_id) == ModelStoreType.FULL_MODEL:
            result = super().recover_model(model_id, execute_checks)
        else:
            # if there is a base model we first have to restore the base model to continue training base on it
            log_rec_base = log_start(self.logging, PROVENANCE, 'recover_model', 'recover_base_model')

            base_model_store_type = self._get_store_type(base_model_id)
            base_model_info = self._recover_base_model(base_model_id, base_model_store_type, execute_checks)
            base_model = base_model_info.model
            log_stop(self.logging, log_rec_base)

            log_load_info = log_start(self.logging, PROVENANCE, 'recover_model', 'load_model_info')
            with tempfile.TemporaryDirectory() as tmp_path:
                # TODO maybe can be replaced when using FileRef Object
                restore_dir = os.path.join(tmp_path, RESTORE_PATH)
                os.mkdir(restore_dir)

                model_info = ModelInfo.load(model_id, self._file_pers_service, self._dict_pers_service, restore_dir,
                                            load_recursive=True, load_files=True)
                recover_info: ProvenanceRecoverInfo = model_info.recover_info
                log_stop(self.logging, log_load_info)

                log_train = log_start(self.logging, PROVENANCE, 'recover_model', 'train')
                train_service = recover_info.train_info.train_service_wrapper.instance
                train_kwargs = recover_info.train_info.train_kwargs
                train_service.train(base_model, **train_kwargs)
                log_stop(self.logging, log_train)

                # because we trained it here the base_model is the updated version
                restored_model = base_model
                restored_model_info = RestoredModelInfo(model=restored_model)

                if execute_checks:
                    log_check_weights = log_start(self.logging, PROVENANCE, 'recover_model', '_check_weights')
                    self._check_weights(restored_model, model_info)
                    log_stop(self.logging, log_check_weights)
                    log_check_env = log_start(self.logging, PROVENANCE, 'recover_model', '_check_env')
                    self._check_env(model_info)
                    log_stop(self.logging, log_check_env)

                result = restored_model_info

        log_stop(self.logging, log_all)
        return result

    def _save_provenance_model(self, model_save_info):
        log_all = log_start(self.logging, PROVENANCE, '_save_provenance_model', 'all')

        log_build_prov = log_start(self.logging, PROVENANCE, '_save_provenance_model', '_build_prov_model_info')
        model_info = self._build_prov_model_info(model_save_info)
        log_stop(self.logging, log_build_prov)

        log_persist = log_start(self.logging, PROVENANCE, '_save_provenance_model', 'persist')
        model_info_id = model_info.persist(self._file_pers_service, self._dict_pers_service)
        log_stop(self.logging, log_persist)

        log_stop(self.logging, log_all)
        return model_info_id

    def add_weights_hash_info(self, model_id: str, model: torch.nn.Module):
        model_info = ModelInfo.load_placeholder(model_id)
        weights_hash_info = WeightDictMerkleTree.from_state_dict(model.state_dict())

        model_info.add_and_persist_weights_hash_info(weights_hash_info, self._dict_pers_service)

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
            environment=model_save_info.environment
        )
        derived_from = model_save_info.base_model if model_save_info.base_model else None
        model_info = ModelInfo(store_type=ModelStoreType.PROVENANCE, recover_info=prov_recover_info,
                               derived_from_id=derived_from)
        return model_info

    def _recover_base_model(self, base_model_id, base_model_store_type, execute_checks=True):
        if base_model_store_type == ModelStoreType.FULL_MODEL:
            return super().recover_model(model_id=base_model_id, execute_checks=execute_checks)
        elif base_model_store_type == ModelStoreType.PROVENANCE:
            return self.recover_model(model_id=base_model_id, execute_checks=execute_checks)
        else:
            raise NotImplementedError


def _get_weights_hash_info(add_weights_hash_info, model_save_info):
    weights_hash_info = None
    if add_weights_hash_info:
        assert model_save_info.model, "to compute a weights info hash the a model has to be given"
        weights_hash_info = WeightDictMerkleTree.from_state_dict(model_save_info.model.state_dict())
    return weights_hash_info

import abc
import os
import sys
import tempfile
import warnings
from enum import Enum

import torch

from mmlib.deterministic import set_deterministic
from mmlib.equal import state_dict_hash, tensor_hash
from mmlib.persistence import AbstractFilePersistenceService, AbstractDictPersistenceService
from schema.model_info import ModelInfo
from schema.recover_info_t1 import RecoverInfoT1
from schema.recover_val import RecoverVal
from schema.schema_obj import SchemaObjType, SchemaObj
from util.zip import zip_path, unzip

ID = '_id'
MODEL_WEIGHTS = 'model_weights'
NAME = 'name'
MODELS = 'models'


class SaveType(Enum):
    PICKLED_WEIGHTS = '1'
    WEIGHT_UPDATES = '2'
    PROVENANCE = '3'


# Future work, se if it would make sense to use protocol here
class AbstractSaveRecoverService(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def save_model(self, model: torch.nn.Module, code: str, code_name: str, recover_val: bool = False,
                   dummy_input_shape: [int] = None) -> str:
        """
        Saves a model together with the given metadata.
        :param model: The actual model to save as an instance of torch.nn.Module.
        :param code: The path to the code of the model (is needed for recover process).
        :param code_name: The name of the model, i.e. the model constructor (is needed for recover process).
        :param recover_val: Indicates if along with the model itself also information is stored to later validate that
        restoring the model lead to the exact same model. It is checked by comparing the model weights and the inference
        result on dummy input. If this flag is true, a dummy_input_shape has to be provided.
        :param dummy_input_shape: The shape of the dummy input that should be used to produce an inference result.
        :return: Returns the id that was used to store the model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save_version(self, model: torch.nn.Module, base_model_id: str, recover_val: bool = False,
                     dummy_input_shape: [int] = None) -> str:
        """
        Saves a new model version by referring to the base_model.
        :param model: The actual model to save as an instance of torch.nn.Module.
        :param base_model_id: the model id of the base_model.
        :param recover_val: Indicates if along with the model itself also information is stored to later validate that
        restoring the model lead to the exact same model. It is checked by comparing the model weights and the inference
        result on dummy input. If this flag is true, the dummy_input_shape is copied form the base model, if there is no
        recover_val stored for the base model it must be given as a parameter.
        :param dummy_input_shape: The shape of the dummy input that should be used to produce an inference result.
        :return: Returns the ID that was used to store the new model version data in the MongoDB.
        """

    @abc.abstractmethod
    def saved_model_ids(self) -> [str]:
        """Returns list of saved models ids"""
        raise NotImplementedError

    @abc.abstractmethod
    def saved_model_infos(self) -> [dict]:
        """Returns list of saved models infos"""
        raise NotImplementedError

    @abc.abstractmethod
    def recover_model(self, model_id: str, check_recover_val=False) -> torch.nn.Module:
        """
        Recovers a the model identified by the given model id.
        :param model_id: The id to identify the model with.
        :param check_recover_val: The flag that indicates if the recover validation data (if there) is used to validate
        the restored model.
        :return: The recovered model as an object of type torch.nn.Module.
        """

    @abc.abstractmethod
    def model_save_size(self, model_id: str) -> float:
        """
        Calculates and returns the amount of bytes that are used for storing the model.
        :param model_id: The id to identify the model.
        :return: The amount of bytes used to store the model.
        """
        raise NotImplementedError


class SimpleSaveRecoverService(AbstractSaveRecoverService):
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

    def save_model(self, model: torch.nn.Module, code: str, code_name: str, recover_val: bool = False,
                   dummy_input_shape: [int] = None) -> str:
        if recover_val:
            assert dummy_input_shape, 'to store recover_val information a dummy input function needs to be provided'

        rec_val_id = None
        if recover_val:
            rec_val_id = self._save_recover_val(model, dummy_input_shape)

        recover_info_t1 = self._save_model_t1(model, code, code_name, recover_val_id=rec_val_id)
        recover_info_id = self._dict_pers_service.save_dict(recover_info_t1.to_dict(), SchemaObjType.RECOVER_T1.value)

        # TODO(future-work) to implement other fields that are default None
        model_id = self._save_model_info(SaveType.PICKLED_WEIGHTS.value, recover_info_id)

        return model_id

    def save_version(self, model: torch.nn.Module, base_model_id: str, recover_val: bool = False,
                     dummy_input_shape: [int] = None) -> str:

        base_model_info = self._get_model_info(base_model_id)
        base_model_recover_info = self._get_recover_info_t1(base_model_info)

        rec_val_id = None
        if recover_val:
            if not dummy_input_shape:
                assert base_model_recover_info.recover_validation, \
                    'neither recover_val for the base model is stored nor a dummy_input_shape is given'
                rec_val = self._get_recover_val(base_model_recover_info.recover_validation)
                dummy_input_shape = rec_val.dummy_input_shape

            rec_val_id = self._save_recover_val(model, dummy_input_shape)

        # copy fields from previous model that will stay the same
        code_name = base_model_recover_info.code_name

        with tempfile.TemporaryDirectory() as tmp_path:
            code = self._file_pers_service.recover_file(base_model_recover_info.model_code, tmp_path)
            recover_info_t1 = self._save_model_t1(model, code, code_name, recover_val_id=rec_val_id)

        recover_info_id = self._dict_pers_service.save_dict(recover_info_t1.to_dict(), SchemaObjType.RECOVER_T1.value)

        # TODO(future-work) to implement other fields that are default None
        model_id = self._save_model_info(SaveType.PICKLED_WEIGHTS.value, recover_info_id, derived_from=base_model_id)

        return model_id

    def saved_model_ids(self) -> [str]:
        return self._dict_pers_service.all_ids_for_type(SchemaObjType.MODEL_INFO.value)

    def saved_model_infos(self) -> [dict]:
        model_ids = self.saved_model_ids()
        return [self._get_model_info(i) for i in model_ids]

    def model_save_size(self, model_id: str) -> float:
        model_info = self._get_model_info(model_id)
        return self._get_save_size(model_info)

    def recover_model(self, model_id: str, check_recover_val=False) -> torch.nn.Module:
        model_info = self._get_model_info(model_id)
        recover_info_t1 = self._get_recover_info_t1(model_info)
        weights_file_id = recover_info_t1.weights

        with tempfile.TemporaryDirectory() as tmp_path:
            code_id = recover_info_t1.model_code
            code = self._file_pers_service.recover_file(code_id, tmp_path)
            generate_call = recover_info_t1.code_name
            model = self._init_model(code, generate_call)

            weights_file = self._file_pers_service.recover_file(weights_file_id, tmp_path)
            s_dict = self._recover_pickled_weights(weights_file, tmp_path)
            model.load_state_dict(s_dict)

        if check_recover_val:
            if recover_info_t1.recover_validation is None:
                warnings.warn('check recoverVal not possible - no recover validation info available')
            else:
                rec_val = self._get_recover_val(recover_info_t1.recover_validation)

                weights_hash = state_dict_hash(model.state_dict())
                assert weights_hash == rec_val.weights_hash, 'check weight hash failed'

                inp_shape = rec_val.dummy_input_shape
                inference_hash = self._get_inference_hash(model, inp_shape)
                assert inference_hash == rec_val.inference_hash, 'check inference hash failed'

        return model

    def _get_save_size(self, obj: SchemaObj):
        if isinstance(obj, ModelInfo):
            return self._get_model_info_size(obj)
        elif isinstance(obj, RecoverInfoT1):
            return self._get_rec_info_t1_size(obj)
        elif isinstance(obj, RecoverVal):
            return self._get_rec_val_size(obj)
        else:
            assert False, 'not implemented'

    def _get_model_info_size(self, model_info: ModelInfo):
        total_size = 0
        # add size of the metadata (dict) itself
        total_size += self._dict_pers_service.dict_size(model_info.m_id, SchemaObjType.MODEL_INFO.value)
        # add size of all references
        model_info_dict = self._get_model_info(model_info.m_id).to_dict()
        # TODO(future-work) keep in mind that we dont want to include the size of "derived from"

        total_size += self._ref_objects_size(model_info)

        return total_size

    def _ref_objects_size(self, root_object):
        total_size = 0
        info_dict = root_object.to_dict()
        for k, v in info_dict.items():
            k_type = root_object.get_type(k)
            if not k_type == SchemaObjType.STRING:
                if k_type == SchemaObjType.FILE:
                    total_size += self._file_pers_service.file_size(v)
                else:
                    schema_obj = self._recover_schema_obj(v, k_type)
                    total_size += self._get_save_size(schema_obj)

        return total_size

    def _get_rec_info_t1_size(self, rec_info: RecoverInfoT1):
        total_size = 0
        # add size of the metadata (dict) itself
        total_size += self._dict_pers_service.dict_size(rec_info.r_id, SchemaObjType.RECOVER_T1.value)
        # add size of all references
        total_size += self._ref_objects_size(rec_info)

        return total_size

    def _get_rec_val_size(self, rec_val: RecoverVal):
        total_size = 0
        # add size of the metadata (dict) itself, RecoVal does not contain any references
        total_size += self._dict_pers_service.dict_size(rec_val.r_id, SchemaObjType.RECOVER_VAL.value)

        return total_size

    def _save_model_info(self, store_type, recover_info_id, derived_from=None, inference_info=None, train_info=None):
        gen_id = self._dict_pers_service.generate_id()
        model_info = ModelInfo(m_id=gen_id, store_type=store_type, recover_info=recover_info_id,
                               derived_from=derived_from, inference_info=inference_info, train_info=train_info)
        model_id = self._dict_pers_service.save_dict(model_info.to_dict(), SchemaObjType.MODEL_INFO.value)
        return model_id

    def _save_model_t1(self, model, code, code_name, recover_val_id=None):
        gen_id = self._dict_pers_service.generate_id()
        with tempfile.TemporaryDirectory() as tmp_path:
            zip_file = self._pickle_weights(model, tmp_path)
            zip_file_id = self._file_pers_service.save_file(zip_file)
            code_file_id = self._file_pers_service.save_file(code)

        recover_info_t1 = RecoverInfoT1(r_id=gen_id, weights=zip_file_id, model_code=code_file_id, code_name=code_name,
                                        recover_validation=recover_val_id)

        return recover_info_t1

    def _get_model_info(self, model_id):
        return self._recover_schema_obj(model_id, SchemaObjType.MODEL_INFO)

    def _get_recover_info_t1(self, model_info):
        recover_info_id = model_info.recover_info
        return self._recover_schema_obj(recover_info_id, SchemaObjType.RECOVER_T1)

    def _get_recover_val(self, recover_val_id):
        return self._recover_schema_obj(recover_val_id, SchemaObjType.RECOVER_VAL)

    def _recover_schema_obj(self, obj_id: str, obj_type: SchemaObjType):
        s_obj = eval('{}()'.format(obj_type.value))
        state_dict = self._dict_pers_service.recover_dict(obj_id, obj_type.value)
        s_obj.load_dict(state_dict)

        return s_obj

    def _pickle_weights(self, model, save_path):
        # store pickle dump of model
        torch.save(model.state_dict(), os.path.join(save_path, MODEL_WEIGHTS))

        # zip everything
        return zip_path(save_path)

    def _recover_pickled_weights(self, weights_file, extract_path):
        unpacked_path = unzip(weights_file, extract_path)
        pickle_path = os.path.join(unpacked_path, MODEL_WEIGHTS)
        state_dict = torch.load(pickle_path)

        return state_dict

    def _init_model(self, code, generate_call):
        path, file = os.path.split(code)
        module = file.replace('.py', '')
        sys.path.append(path)
        exec('from {} import {}'.format(module, generate_call))
        model = eval('{}()'.format(generate_call))

        return model

    def _save_recover_val(self, model, dummy_input_shape):
        gen_id = self._dict_pers_service.generate_id()
        weights_hash = state_dict_hash(model.state_dict())

        inference_hash = self._get_inference_hash(model, dummy_input_shape)

        recover_val = RecoverVal(r_id=gen_id, weights_hash=weights_hash, inference_hash=inference_hash,
                                 dummy_input_shape=dummy_input_shape)

        recover_val_id = self._dict_pers_service.save_dict(recover_val.to_dict(), SchemaObjType.RECOVER_VAL.value)

        return recover_val_id

    def _get_inference_hash(self, model, dummy_input_shape):
        set_deterministic()
        model.eval()
        dummy_input = torch.rand(dummy_input_shape)
        dummy_output = model(dummy_input)
        inference_hash = tensor_hash(dummy_output)
        return inference_hash

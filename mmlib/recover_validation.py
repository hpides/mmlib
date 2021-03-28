import warnings

from mmlib.persistence import AbstractDictPersistenceService
from util.hash import state_dict_hash, inference_hash

ID = 'id'
WEIGHTS_HASH = 'weights_hash'
INFERENCE_HASH = 'inference_hash'
DUMMY_INPUT_SHAPE = 'dummy_input_shape'

RECOVER_VAL = 'recover_val'


class RecoverVal:

    def __init__(self, weights_hash: str, inference_hash: str, dummy_input_shape: [int]):
        self.weights_hash = weights_hash
        self.inference_hash = inference_hash
        self.dummy_input_shape = dummy_input_shape

    def to_dict(self) -> dict:
        dict_representation = {
            WEIGHTS_HASH: self.weights_hash,
            INFERENCE_HASH: self.inference_hash,
            DUMMY_INPUT_SHAPE: self.dummy_input_shape,
        }

        return dict_representation

    @classmethod
    def restore_from_dict(cls, state_dict):
        weights_hash = state_dict[WEIGHTS_HASH]
        inference_hash = state_dict[INFERENCE_HASH]
        dummy_input_shape = state_dict[DUMMY_INPUT_SHAPE]

        return cls(weights_hash=weights_hash, inference_hash=inference_hash, dummy_input_shape=dummy_input_shape)


class RecoverValidationService:

    def __init__(self, dict_save_service: AbstractDictPersistenceService):
        self.dict_save_service = dict_save_service

    def save_recover_val_info(self, model, model_id, dummy_input_shape):
        recover_val = self._generate_recover_val(model, dummy_input_shape)
        store_dict = recover_val.to_dict()

        # use model_id as id
        store_dict[ID] = model_id

        self.dict_save_service.save_dict(store_dict, RECOVER_VAL)

    def _generate_recover_val(self, model, dummy_input_shape):
        assert model is not None, 'if recover val should be generated, then model needs to be given'

        weights_hash = state_dict_hash(model.state_dict())
        inf_hash = inference_hash(model, dummy_input_shape)

        recover_val = RecoverVal(weights_hash=weights_hash, inference_hash=inf_hash,
                                 dummy_input_shape=dummy_input_shape)

        return recover_val

    def _check_recover_val(self, model_id, model):
        if not self.dict_save_service.id_exists(model_id, RECOVER_VAL):
            warnings.warn('check recoverVal not possible - no recover validation info available')
            return True

        restored_dict = self.dict_save_service.recover_dict(model_id, RECOVER_VAL)
        recover_val = RecoverVal.restore_from_dict(restored_dict)
        weights_hash = state_dict_hash(model.state_dict())
        inp_shape = recover_val.dummy_input_shape
        inf_hash = inference_hash(model, inp_shape)

        return weights_hash == recover_val.weights_hash and inf_hash == recover_val.inference_hash

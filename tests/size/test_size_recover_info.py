import os
import tempfile

import torch

from mmlib.track_env import track_current_environment
from schema.file_reference import FileReference
from schema.recover_info import FullModelRecoverInfo, ENVIRONMENT, MODEL_CODE, PARAMETERS, WeightsUpdateRecoverInfo, \
    UPDATE
from schema.schema_obj import METADATA_SIZE
from tests.example_files.mynets.resnet18 import resnet18
from tests.size.abstract_test_size import TestSize

FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class TestRecoverInfoSize(TestSize):

    def test_full_model_size(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            model = resnet18(pretrained=True)
            param_path = os.path.join(tmp_path, 'params')
            torch.save(model.state_dict(), param_path)
            environment = track_current_environment()
            recover_info = FullModelRecoverInfo(
                parameters_file=FileReference(param_path),
                model_code=FileReference(os.path.join(FILE_PATH, '../example_files/mynets/resnet18.py')),
                environment=environment)
            env_id = recover_info.persist(self.file_pers_service, self.dict_pers_service)

        place_holder = FullModelRecoverInfo.load_placeholder(env_id)
        size_dict = place_holder.size_info(self.file_pers_service, self.dict_pers_service)

        # raw data number from mac finder info
        self.assertTrue(METADATA_SIZE in size_dict.keys())
        # we expect four fields: metadata, environment, model code, model parameters
        self.assertEqual(4, len(size_dict.keys()))
        self.assertTrue(size_dict[METADATA_SIZE] > 0)
        self.assertTrue(size_dict[ENVIRONMENT][METADATA_SIZE] > 0)
        self.assertTrue(size_dict[MODEL_CODE] > 0)
        self.assertTrue(size_dict[PARAMETERS] > size_dict[MODEL_CODE])

    def test_param_update_size(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            model = resnet18(pretrained=True)
            update_path = os.path.join(tmp_path, 'update')
            torch.save(model.state_dict(), update_path)
            recover_info = WeightsUpdateRecoverInfo(
                update=FileReference(update_path)
            )
            env_id = recover_info.persist(self.file_pers_service, self.dict_pers_service)

        place_holder = WeightsUpdateRecoverInfo.load_placeholder(env_id)
        size_dict = place_holder.size_info(self.file_pers_service, self.dict_pers_service)

        # raw data number from mac finder info
        self.assertTrue(METADATA_SIZE in size_dict.keys())
        # we expect four fields: metadata, environment, model code, model parameters
        self.assertEqual(2, len(size_dict.keys()))
        self.assertTrue(size_dict[METADATA_SIZE] > 0)
        self.assertTrue(size_dict[UPDATE] > 0)

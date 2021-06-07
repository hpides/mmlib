import os
import tempfile

import torch

from mmlib.schema.file_reference import FileReference
from mmlib.schema.model_info import ModelInfo
from mmlib.schema.recover_info import FullModelRecoverInfo, RECOVER_INFO
from mmlib.schema.schema_obj import METADATA_SIZE
from mmlib.schema.store_type import ModelStoreType
from mmlib.track_env import track_current_environment
from mmlib.util.weight_dict_merkle_tree import WeightDictMerkleTree
from tests.example_files.mynets.resnet18 import resnet18
from tests.size.abstract_test_size import TestSize

FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class TestModelInfoSize(TestSize):

    def test_environment_size(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            model = resnet18(pretrained=True)
            param_path = os.path.join(tmp_path, 'params')
            torch.save(model.state_dict(), param_path)
            environment = track_current_environment()
            recover_info = FullModelRecoverInfo(
                parameters_file=FileReference(param_path),
                model_code=FileReference(os.path.join(FILE_PATH, '../example_files/mynets/resnet18.py')),
                environment=environment)

            model_info = ModelInfo(
                store_type=ModelStoreType.FULL_MODEL,
                recover_info=recover_info,
                weights_hash_info=WeightDictMerkleTree.from_state_dict(model.state_dict())

            )

            _id = model_info.persist(self.file_pers_service, self.dict_pers_service)

        place_holder = ModelInfo.load_placeholder(_id)
        size_dict = place_holder.size_info(self.file_pers_service, self.dict_pers_service)

        self.assertEqual(len(size_dict.keys()), 2)
        self.assertTrue(size_dict[METADATA_SIZE] > 0)
        self.assertTrue(size_dict[RECOVER_INFO][METADATA_SIZE] > 0)

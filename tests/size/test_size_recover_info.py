import os
import tempfile

import torch

from mmlib.track_env import track_current_environment
from schema.dataset import Dataset
from schema.file_reference import FileReference
from schema.recover_info import FullModelRecoverInfo, ENVIRONMENT, MODEL_CODE, PARAMETERS, WeightsUpdateRecoverInfo, \
    UPDATE, ProvenanceRecoverInfo, DATASET, TRAIN_INFO
from schema.restorable_object import StateFileRestorableObjectWrapper
from schema.schema_obj import METADATA_SIZE
from schema.train_info import TrainInfo
from tests.example_files.imagenet_optimizer import ImagenetOptimizer
from tests.example_files.imagenet_train import ImagenetTrainService, OPTIMIZER, ImagenetTrainWrapper
from tests.example_files.mynets.resnet18 import resnet18
from tests.size.abstract_test_size import TestSize

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
COCO_DATA = os.path.join(FILE_PATH, '../example_files/data/reduced-custom-coco-data')
IMG_TRAIN_WRAPPER_CODE = os.path.join(FILE_PATH, '../example_files/imagenet_train.py')
OPTIMIZER_CODE = os.path.join(FILE_PATH, '../example_files/imagenet_optimizer.py')


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
            _id = recover_info.persist(self.file_pers_service, self.dict_pers_service)

        place_holder = FullModelRecoverInfo.load_placeholder(_id)
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
            _id = recover_info.persist(self.file_pers_service, self.dict_pers_service)

        place_holder = WeightsUpdateRecoverInfo.load_placeholder(_id)
        size_dict = place_holder.size_info(self.file_pers_service, self.dict_pers_service)

        # raw data number from mac finder info
        self.assertTrue(METADATA_SIZE in size_dict.keys())
        # we expect four fields: metadata, environment, model code, model parameters
        self.assertEqual(2, len(size_dict.keys()))
        self.assertTrue(size_dict[METADATA_SIZE] > 0)
        self.assertTrue(size_dict[UPDATE] > 0)

    def test_provenance_size(self):
        file = FileReference(path=COCO_DATA)
        data_set = Dataset(raw_data=file)
        environment = track_current_environment()
        train_info = self._dummy_train_service()
        recover_info = ProvenanceRecoverInfo(
            dataset=data_set,
            environment=environment,
            train_info=train_info
        )
        _id = recover_info.persist(self.file_pers_service, self.dict_pers_service)

        place_holder = ProvenanceRecoverInfo.load_placeholder(_id)
        size_dict = place_holder.size_info(self.file_pers_service, self.dict_pers_service)

        # raw data number from mac finder info
        self.assertTrue(METADATA_SIZE in size_dict.keys())
        # we expect four fields: metadata, dataset, train info, environment
        self.assertEqual(4, len(size_dict.keys()))
        self.assertTrue(size_dict[METADATA_SIZE] > 0)
        self.assertTrue(size_dict[ENVIRONMENT][METADATA_SIZE] > 0)
        self.assertTrue(size_dict[DATASET][METADATA_SIZE] > 0)
        self.assertTrue(size_dict[TRAIN_INFO][METADATA_SIZE] > 0)

    def _dummy_train_service(self):
        train_info = TrainInfo(
            ts_wrapper=self._dummy_train_service_wrapper(),
            ts_wrapper_class_name='ImagenetTrainWrapper',
            ts_wrapper_code=FileReference(IMG_TRAIN_WRAPPER_CODE),
            train_kwargs={}
        )

        return train_info

    def _dummy_train_service_wrapper(self):
        model = resnet18()
        imagenet_ts = ImagenetTrainService()
        state_dict = {}

        # for dummy put optimizer in
        optimizer_kwargs = {'lr': 1e-4, 'momentum': 0.9, 'weight_decay': 1e-4}
        optimizer = ImagenetOptimizer(model.parameters(), **optimizer_kwargs)
        state_dict[OPTIMIZER] = StateFileRestorableObjectWrapper(
            code=FileReference(OPTIMIZER_CODE),
            init_args=optimizer_kwargs,
            init_ref_type_args=['params'],
            instance=optimizer
        )

        imagenet_ts.state_objs = state_dict

        return ImagenetTrainWrapper(instance=imagenet_ts)

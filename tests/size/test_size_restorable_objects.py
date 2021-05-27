import os
import shutil

import torch

from schema.restorable_object import RestorableObjectWrapper, CODE_FILE, OptimizerWrapper, STATE_FILE
from schema.schema_obj import METADATA_SIZE
from tests.example_files.data.custom_coco import TrainCustomCoco
from tests.example_files.mynets.resnet18 import resnet18
from tests.size.abstract_test_size import TestSize

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
COCO_DATA = '../example_files/data/reduced-custom-coco-data'

MONGO_CONTAINER_NAME = 'mongo-test'


class TestRestorableObjectSize(TestSize):

    def __clean_up(self):
        os.system('docker kill %s' % MONGO_CONTAINER_NAME)
        if os.path.exists(self.abs_tmp_path):
            shutil.rmtree(self.abs_tmp_path)

    def test_restorable_object_wrapper_size_code(self):
        instance = TrainCustomCoco(os.path.join(FILE_PATH, COCO_DATA))
        row = RestorableObjectWrapper(
            config_args={'root': ''},
            instance=instance
        )

        row_id = row.persist(self.file_pers_service, self.dict_pers_service)

        place_holder = RestorableObjectWrapper.load_placeholder(row_id)
        size_dict = place_holder.size_info(self.file_pers_service, self.dict_pers_service)

        # raw data number from mac finder info
        self.assertEqual(len(size_dict.keys()), 2)
        self.assertTrue(METADATA_SIZE in size_dict.keys())
        self.assertTrue(CODE_FILE in size_dict.keys())
        self.assertTrue(size_dict[METADATA_SIZE] > 0)
        # code should be bigger than 3 kB
        self.assertTrue(size_dict[CODE_FILE] > 3000)

    def test_restorable_object_wrapper_size_import(self):
        data_loader_kwargs = {'batch_size': 5, 'shuffle': True, 'num_workers': 0, 'pin_memory': True}
        dataloader = TrainCustomCoco(os.path.join(FILE_PATH, COCO_DATA))
        row = RestorableObjectWrapper(
            import_cmd='from torch.utils.data import DataLoader',
            init_args=data_loader_kwargs,
            init_ref_type_args=['dataset'],
            instance=dataloader
        )

        row_id = row.persist(self.file_pers_service, self.dict_pers_service)

        place_holder = RestorableObjectWrapper.load_placeholder(row_id)
        size_dict = place_holder.size_info(self.file_pers_service, self.dict_pers_service)

        # raw data number from mac finder info
        self.assertEqual(len(size_dict.keys()), 1)
        self.assertTrue(METADATA_SIZE in size_dict.keys())
        self.assertTrue(size_dict[METADATA_SIZE] > 0)

    def test_state_file_restorable_object_wrapper_size_import(self):
        model = resnet18()
        optimizer_kwargs = {'lr': 1e-4, 'momentum': 0.9, 'weight_decay': 1e-4}
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_kwargs)
        row = OptimizerWrapper(
            import_cmd='from torch.optim import SGD',
            init_args=optimizer_kwargs,
            init_ref_type_args=['params'],
            instance=optimizer
        )

        row_id = row.persist(self.file_pers_service, self.dict_pers_service)

        place_holder = OptimizerWrapper.load_placeholder(row_id)
        size_dict = place_holder.size_info(self.file_pers_service, self.dict_pers_service)

        # raw data number from mac finder info
        self.assertEqual(len(size_dict.keys()), 2)
        self.assertTrue(METADATA_SIZE in size_dict.keys())
        self.assertTrue(size_dict[METADATA_SIZE] > 0)
        # state should be bigger than 0
        self.assertTrue(size_dict[STATE_FILE] > 0)

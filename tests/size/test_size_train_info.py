import os

from mmlib.schema import FileReference
from mmlib.schema import METADATA_SIZE
from mmlib.schema.train_info import TrainInfo, WRAPPER_CODE, TRAIN_SERVICE
from tests.size.abstract_test_size import TestSize
from tests.size.test_size_restorable_objects import _get_dummy_imagenet_train_service_wrapper

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
TRAIN_WRAPPER_CODE = os.path.join(FILE_PATH, '../example_files/imagenet_train.py')

MONGO_CONTAINER_NAME = 'mongo-test'


class TestTrainInfoSize(TestSize):

    def test_train_info_size(self):
        ts_wrapper = _get_dummy_imagenet_train_service_wrapper()
        train_info = TrainInfo(
            ts_wrapper=ts_wrapper,
            ts_wrapper_code=FileReference(TRAIN_WRAPPER_CODE),
            ts_wrapper_class_name='ImagenetTrainWrapper',
            train_kwargs={'dummy': 'dummy'}
        )

        _id = train_info.persist(self.file_pers_service, self.dict_pers_service)

        place_holder = TrainInfo.load_placeholder(_id)
        size_dict = place_holder.size_info(self.file_pers_service, self.dict_pers_service)

        # raw data number from mac finder info
        self.assertEqual(len(size_dict.keys()), 3)
        self.assertTrue(size_dict[METADATA_SIZE] > 0)
        self.assertTrue(size_dict[WRAPPER_CODE] > 0)
        self.assertTrue(size_dict[TRAIN_SERVICE][METADATA_SIZE] > 0)

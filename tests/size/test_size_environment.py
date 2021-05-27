from mmlib.track_env import track_current_environment
from schema.dataset import Dataset
from schema.environment import Environment
from schema.file_reference import FileReference
from tests.size.abstract_test_size import TestSize

COCO_DATA = '../example_files/data/reduced-custom-coco-data'

MONGO_CONTAINER_NAME = 'mongo-test'


class TestDatasetSize(TestSize):

    def test_environment_size(self):
        environment = track_current_environment()
        env_id = environment.persist(self.file_pers_service, self.dict_pers_service)

        place_holder = Environment.load_placeholder(env_id)
        size_dict = place_holder.size_info(self.file_pers_service, self.dict_pers_service)

        # raw data number from mac finder info
        self.assertTrue('metadata_size' in size_dict.keys())
        self.assertEqual(len(size_dict.keys()), 1)
        self.assertTrue(size_dict['metadata_size'] > 0)

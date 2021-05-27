from mmlib.track_env import track_current_environment
from schema.environment import Environment
from schema.schema_obj import METADATA_SIZE
from tests.size.abstract_test_size import TestSize


class TestEnvSize(TestSize):

    def test_environment_size(self):
        environment = track_current_environment()
        env_id = environment.persist(self.file_pers_service, self.dict_pers_service)

        place_holder = Environment.load_placeholder(env_id)
        size_dict = place_holder.size_info(self.file_pers_service, self.dict_pers_service)

        # raw data number from mac finder info
        self.assertTrue(METADATA_SIZE in size_dict.keys())
        self.assertEqual(len(size_dict.keys()), 1)
        self.assertTrue(size_dict[METADATA_SIZE] > 0)

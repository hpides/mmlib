from mmlib.save import WeightUpdateSaveService
from tests.test_baseline_save_servcie import TestBaselineSaveService


class TestWeightUpdateSaveService(TestBaselineSaveService):

    def init_save_service(self, dict_pers_service, file_pers_service):
        self.save_service = WeightUpdateSaveService(file_pers_service, dict_pers_service)

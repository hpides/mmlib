from mmlib.save import WeightUpdateSaveService
from tests.save.test_weight_update_save_service import TestWeightUpdateSaveService


class TestImprovedWeightUpdateSaveService(TestWeightUpdateSaveService):

    def init_save_service(self, dict_pers_service, file_pers_service):
        self.save_service = WeightUpdateSaveService(file_pers_service, dict_pers_service, improved_version=True)

from mmlib.equal import model_equal
from mmlib.save import WeightUpdateSaveService
from schema.save_info_builder import ModelSaveInfoBuilder
from tests.example_files.mynets.resnet18 import resnet18
from tests.save.test_baseline_save_servcie import TestBaselineSaveService, NETWORK_CODE_TEMPLATE, RESNET_18
from util.dummy_data import imagenet_input


class TestWeightUpdateSaveService(TestBaselineSaveService):

    def init_save_service(self, dict_pers_service, file_pers_service):
        self.save_service = WeightUpdateSaveService(file_pers_service, dict_pers_service)

    def test_save_restore_many_derived_models(self):
        code_file = NETWORK_CODE_TEMPLATE.format(RESNET_18)
        initial_model = resnet18()

        # save initial model
        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(initial_model, code_file)
        save_info = save_info_builder.build()
        initial_model_id = self.save_service.save_model(save_info)

        # save derived model
        derived_model = resnet18(pretrained=True)
        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(derived_model, code_file, base_model_id=initial_model_id)
        save_info = save_info_builder.build()
        derived_model_id = self.save_service.save_model(save_info)

        restored_model_info = self.save_service.recover_model(derived_model_id)

        self.assertTrue(model_equal(derived_model, restored_model_info.model, imagenet_input))

        # save derived model
        derived_model_2 = resnet18()
        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(derived_model_2, code_file, base_model_id=derived_model_id)
        save_info = save_info_builder.build()
        derived_model_id_2 = self.save_service.save_model(save_info)

        restored_model_info_2 = self.save_service.recover_model(derived_model_id_2)

        self.assertTrue(model_equal(derived_model_2, restored_model_info_2.model, imagenet_input))

        # save derived model
        derived_model_3 = resnet18()
        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(derived_model_3, code_file, base_model_id=derived_model_id_2)
        save_info = save_info_builder.build()
        derived_model_id_3 = self.save_service.save_model(save_info)

        restored_model_info_3 = self.save_service.recover_model(derived_model_id_3)

        self.assertTrue(model_equal(derived_model_3, restored_model_info_3.model, imagenet_input))

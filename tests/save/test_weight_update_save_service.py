import tempfile

from mmlib.equal import model_equal
from mmlib.save import WeightUpdateSaveService
from schema.model_info import ModelInfo
from schema.recover_info import WeightsUpdateRecoverInfo, FullModelRecoverInfo
from schema.save_info_builder import ModelSaveInfoBuilder
from tests.example_files.mynets.resnet18 import resnet18
from tests.save.test_baseline_save_servcie import TestBaselineSaveService
from util.dummy_data import imagenet_input


class TestWeightUpdateSaveService(TestBaselineSaveService):

    def init_save_service(self, dict_pers_service, file_pers_service):
        self.save_service = WeightUpdateSaveService(file_pers_service, dict_pers_service)

    def test_save_restore_many_derived_models(self):
        initial_model = resnet18()

        # save initial model
        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(model=initial_model)
        save_info = save_info_builder.build()
        initial_model_id = self.save_service.save_model(save_info)

        # save derived model
        derived_model = resnet18(pretrained=True)
        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(model=derived_model, base_model_id=initial_model_id)
        save_info = save_info_builder.build()
        derived_model_id = self.save_service.save_model(save_info)

        restored_model_info = self.save_service.recover_model(derived_model_id)

        self.assertTrue(model_equal(derived_model, restored_model_info.model, imagenet_input))

        # save derived model
        derived_model_2 = restored_model_info.model
        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(model=derived_model_2, base_model_id=derived_model_id)
        save_info = save_info_builder.build()
        derived_model_id_2 = self.save_service.save_model(save_info)

        restored_model_info_2 = self.save_service.recover_model(derived_model_id_2)

        self.assertTrue(model_equal(derived_model_2, restored_model_info_2.model, imagenet_input))

        # save derived model
        derived_model_3 = restored_model_info_2.model
        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(model=derived_model_3, base_model_id=derived_model_id_2)
        save_info = save_info_builder.build()
        derived_model_id_3 = self.save_service.save_model(save_info)

        restored_model_info_3 = self.save_service.recover_model(derived_model_id_3)

        self.assertTrue(model_equal(derived_model_3, restored_model_info_3.model, imagenet_input))

    def test_patch_size_almost_zero(self):
        initial_model = resnet18(pretrained=True)
        # save initial model
        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(model=initial_model)
        save_info = save_info_builder.build()
        initial_model_id = self.save_service.save_model(save_info)

        # save derived model
        derived_model = resnet18(pretrained=True)
        save_info_builder = ModelSaveInfoBuilder()
        save_info_builder.add_model_info(model=derived_model, base_model_id=initial_model_id)
        save_info = save_info_builder.build()
        derived_model_id = self.save_service.save_model(save_info)

        restored_model_info = self.save_service.recover_model(derived_model_id)

        # the models should all be the same
        self.assertTrue(model_equal(derived_model, restored_model_info.model, imagenet_input))
        self.assertTrue(model_equal(initial_model, restored_model_info.model, imagenet_input))
        self.assertTrue(model_equal(initial_model, derived_model, imagenet_input))

        # since they are the same the patch should be almost zero
        with tempfile.TemporaryDirectory() as tmp_path:
            initial_model_info = ModelInfo.load(initial_model_id, self.file_pers_service, self.dict_pers_service,
                                                tmp_path, load_recursive=True)
            derived_model_info = ModelInfo.load(derived_model_id, self.file_pers_service, self.dict_pers_service,
                                                tmp_path, load_recursive=True)

            initial_recover_info: FullModelRecoverInfo = initial_model_info.recover_info
            derived_recover_info: WeightsUpdateRecoverInfo = derived_model_info.recover_info

            initial_weights_file = initial_recover_info.weights_file
            self.file_pers_service.file_size(initial_weights_file)

            weights_update_file = derived_recover_info.update
            self.file_pers_service.file_size(weights_update_file)

            self.assertTrue(initial_weights_file.size > weights_update_file.size)
            self.assertTrue(weights_update_file.size, 10000)

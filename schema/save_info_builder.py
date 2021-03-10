import torch

from mmlib.save_info import ModelSaveInfo


class ModelSaveInfoBuilder:

    def __init__(self):
        super().__init__()
        self._model = None
        self._base_model = None
        self._code = None
        self._code_name = None
        self._recover_val = False
        self._dummy_input_shape = None

    def add_model_info(self, model: torch.nn.Module, code: str = None, model_class_name: str = None,
                       base_model: str = None):
        """
        Adds the general model information
        :param model: The actual model to save as an instance of torch.nn.Module.
        :param code: (only required if base model not given) The path to the code of the model
        (is needed for recover process).
        :param model_class_name: (only required if base model not given) The name of the model, i.e. the model constructor (is needed for recover process).
        :param base_model: The id of the base model.
        """
        self._model = model
        self._base_model = base_model
        self._code = code
        self._code_name = model_class_name

    def add_recover_val(self, dummy_input_shape: [int] = None):
        """
        Indicates that recover validation info should be saved and adds the required info.
        :param dummy_input_shape: The shape of the dummy input that should be used to produce an inference result.
        """
        self._recover_val = True
        self._dummy_input_shape = dummy_input_shape

    def build(self) -> ModelSaveInfo:
        # TODO check if all info is available
        save_info = ModelSaveInfo(self._model, self._base_model, self._code, self._code_name, self._recover_val,
                                  self._dummy_input_shape)
        return save_info


# class FullModelVersionSafeInfoBuilder(RecoverValInfoBuilder):
#
#     def __init__(self):
#         super().__init__()
#         self._model = None
#         self._base_model_id = None
#
#     def add_model_version_info(self, model: torch.nn.Module, base_model_id: str):
#         """
#         :param model: The actual model to save as an instance of torch.nn.Module.
#         :param base_model_id: the model id of the base_model.
#         """
#         self._model = model
#         self._base_model_id = base_model_id
#
#     def build(self) -> FullModelVersionSaveInfo:
#         version_info = FullModelVersionSaveInfo(self._model, self._base_model_id, self._recover_val,
#                                                 self._dummy_input_shape)
#         return version_info
#
#
# class DataLoaderSaveInfo:
#
#     def __init__(self):
#         self.data_loader_code = None
#         self.data_loader_class_name = None
#         self.data_loader_state = None
#
#     def add_data_loader_info(self, code: str, class_name: str, state: str):
#         self.data_loader_code = code
#         self.data_loader_class_name = class_name
#         self.data_loader_state = state
#
#
# class PreProcessorSaveInfo:
#
#     def __init__(self):
#         self.pre_processor_code = None
#         self.pre_processor_class_name = None
#         self.pre_processor_state = None
#
#     def add_data_loader_info(self, code: str, class_name: str, state: str):
#         self.pre_processor_code = code
#         self.pre_processor_class_name = class_name
#         self.pre_processor_state = state
#
#
# class DatasetSaveInfo:
#
#     def __init__(self):
#         self.dataset_code = None
#         self.dataset_class_name = None
#         self.dataset_raw_data = None
#         self.dataset_raw_data_size = None
#
#     def add_dataset_save_info(self, dataset_code: str, dataset_class_name: str, dataset_raw_data: str,
#                               dataset_raw_data_size: str):
#         self.dataset_code = dataset_code
#         self.dataset_class_name = dataset_class_name
#         self.dataset_raw_data = dataset_raw_data
#         self.dataset_raw_data_size = dataset_raw_data_size
#
#
# class TrainInfoSaveInfo(DataLoaderSaveInfo, PreProcessorSaveInfo, DatasetSaveInfo):
#
#     def __init__(self):
#         DataLoaderSaveInfo.__init__(self)
#         PreProcessorSaveInfo.__init__(self)
#         DatasetSaveInfo.__init__(self)
#

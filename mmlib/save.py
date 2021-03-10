import abc

from mmlib.save_info import ModelSaveInfo, ModelRestoreInfo


# Future work, se if it would make sense to use protocol here
class AbstractSaveService(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def save_model(self, model_safe_info: ModelSaveInfo) -> str:
        """
        Saves a model together with the given metadata.
        :param model_safe_info: An instance of ModelSaveInfo providing all the info needed to save the model.
        :return: Returns the id that was used to store the model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def recover_model(self, model_id: str, check_recover_val=False) -> ModelRestoreInfo:
        """
        Recovers a the model and metadata identified by the given model id.
        :param model_id: The id to identify the model with.
        :param check_recover_val: The flag that indicates if the recover validation data (if there) is used to validate
        the restored model.
        :return: The recovered model and metadata bundled in an object of type ModelRestoreInfo.
        """

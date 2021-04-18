import os

from mmlib.equal import model_equal
from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from mmlib.save import BaselineSaveService
from schema.save_info_builder import ModelSaveInfoBuilder
from tests.networks.mynets.mobilenet import mobilenet_v2
from util.dummy_data import imagenet_input

CONTAINER_NAME = 'mongo-test'

TARGET_FILE_SYSTEM_DIR = './filesystem-tmp'

if __name__ == '__main__':
    # initialize a service to store files
    abs_tmp_path = os.path.abspath(TARGET_FILE_SYSTEM_DIR)
    file_pers_service = FileSystemPersistenceService(abs_tmp_path)
    # run mongoDB locally in docker container and initialize service to store dictionaries (JSON)
    os.system('docker run --rm --name %s -it -p 27017:27017 -d  mongo:4.4.3 ' % CONTAINER_NAME)
    dict_pers_service = MongoDictPersistenceService()
    # initialize baseline save service
    save_service = BaselineSaveService(file_pers_service, dict_pers_service)
    # initialize instance of mobilenet_v2
    model = mobilenet_v2(pretrained=True)
    # create the info to save the model
    save_info_builder = ModelSaveInfoBuilder()
    save_info_builder.add_model_info(
        model=model,
        code='../tests/networks/mynets/mobilenet.py',
        class_name='mobilenet_v2')
    save_info = save_info_builder.build()
    # given the save info we can store the model, ad get a model id back
    model_id = save_service.save_model(save_info)
    # having this model id, we can restore the model using the save service
    # it accesses the mongoDB for meta info, and the file system for files (model weights and code)
    restored_model_info = save_service.recover_model(model_id)
    # finally we can check if the model we have stored is equal to the recovered model
    # model_equal compares all model weights, and makes a dummy prediction that is also compared
    if model_equal(model, restored_model_info.model, imagenet_input):
        print('Success: the stored and the restored models are equal!')

    # kill the docker container
    os.system('docker kill %s' % CONTAINER_NAME)

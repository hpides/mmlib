import os
import zipfile

from mmlib.util.helper import zip_dir


def zip_path(save_path: str) -> str:
    """
    Compresses a given directory using pythons zip functionality.
    :param save_path: The directory path that should be zipped.
    :return: The path to the generated .zip file.
    """
    path, name = os.path.split(save_path)
    # temporarily change dict for zip process
    owd = os.getcwd()
    os.chdir(path)
    zip_name = name + '.zip'
    zip_dir(name, zip_name)
    # change path back
    os.chdir(owd)

    return os.path.join(path, zip_name)


def unzip(zip_p, extract_path) -> str:
    """
    Unzips a .zip file using pythons zip functionality.
    :param zip_p: The path to the .zip file.
    :param extract_path: The directory where the files should be extracted to.
    :return: The path to the unzipped files.
    """
    with zipfile.ZipFile(zip_p, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    path, file_name = os.path.split(zip_p)
    unpack_path = os.path.join(extract_path, file_name.split('.')[0])

    return unpack_path

import os
import shutil
from zipfile import ZipFile

import torch
from colorama import Fore, Style


def print_info(message):
    print(Fore.GREEN + "INFO: " + message + Style.RESET_ALL + '\n')


def get_all_file_paths(directory):
    file_paths = []

    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths


def zip_dir(root, dst_path):
    all_files = get_all_file_paths(root)
    with ZipFile(dst_path, 'w') as zip:
        # writing each file one by one
        for file in all_files:
            zip.write(file)


def find_zip_file(path):
    return find_file(path, ending='.zip')


def find_file(path, ending=None):
    for r, d, f in os.walk(path):
        for item in f:
            if ending is None:
                return os.path.join(r, item)
            elif ending in item:
                return os.path.join(r, item)



def get_device(device):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def clean(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)

import inspect
import json
import os
import shutil
import time
import uuid
from zipfile import ZipFile

import torch
from colorama import Fore, Style

TIME = 'time'

STOP = 'stop'

START_STOP = 'start-stop'

START = 'start'


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
    elif os.path.exists(path):
        os.remove(path)


def copy_all_data(src_root, dst_root):
    src_root = os.path.abspath(src_root)
    dst_root = os.path.abspath(dst_root)

    shutil.copytree(src_root, dst_root)


def move_data(src_root, dst_root):
    src_root = os.path.abspath(src_root)
    dst_root = os.path.abspath(dst_root)

    shutil.move(src_root, dst_root)


def class_name(obj: object) -> str:
    return obj.__class__.__name__


def source_file(obj: object) -> str:
    return inspect.getsourcefile(obj.__class__)


def log_start(logging, approach, method, event_key):
    if logging:
        t = time.time_ns()
        _id = uuid.uuid4()
        log_dict = {
            START_STOP: START,
            '_id': str(_id),
            'approach': approach,
            'method': method,
            'event': event_key,
            TIME: t
        }

        print(json.dumps(log_dict))

        return log_dict


def log_stop(logging, log_dict):
    if logging:
        assert log_dict[START_STOP] == START

        t = time.time_ns()

        log_dict[START_STOP] = STOP
        log_dict[TIME] = t

        print(json.dumps(log_dict))

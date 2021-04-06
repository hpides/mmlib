import locale
import platform
import subprocess

import torch
from torch.utils.collect_env import get_pretty_env_info

from schema.environment import Environment

ARCHITECTURE = 'architecture'
MACHINE = 'machine'
NODE = 'node'
PLATFORM = 'platform'
PROCESSOR = 'processor'
PYTHON_BUILD = 'python_build'
PYTHON_COMPILER = 'python_compiler'
PYTHON_BRANCH = 'python_branch'
PYTHON_IMPLEMENTATION = 'python_implementation'
PYTHON_REVISION = 'python revision'
PYTHON_VERSION = 'python_version'
PYTHON_VERSION_TUPLE = 'python_version_tuple'
RELEASE = 'release'
SYSTEM = 'system'
VERSION = 'version'
UNAME = 'uname'
MAC_VER = 'mac_ver'
LIBC_VER = 'libc_ver'


def get_python_platform_info():
    python_env_dict = {
        ARCHITECTURE: platform.architecture(),
        MACHINE: platform.machine(),
        NODE: platform.node(),
        PLATFORM: platform.platform(),
        PROCESSOR: platform.processor(),
        PYTHON_BUILD: platform.python_build(),
        PYTHON_COMPILER: platform.python_compiler(),
        PYTHON_BRANCH: platform.python_branch(),
        PYTHON_IMPLEMENTATION: platform.python_implementation(),
        PYTHON_REVISION: platform.python_revision(),
        PYTHON_VERSION: platform.python_version(),
        PYTHON_VERSION_TUPLE: platform.python_version_tuple(),
        RELEASE: platform.release(),
        SYSTEM: platform.system(),
        VERSION: platform.version(),
        UNAME: platform.uname(),
        MAC_VER: platform.mac_ver(),
        LIBC_VER: platform.libc_ver()
    }

    return python_env_dict


def get_python_libs():
    output: str = _run('pip freeze')[1]
    installed = output.split('\n')
    return installed


def _run(command):
    # copied from torch.utils.collect_env
    """Returns (return-code, stdout, stderr)"""
    p = subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True)
    raw_output, raw_err = p.communicate()
    rc = p.returncode
    enc = locale.getpreferredencoding()
    output = raw_output.decode(enc)
    err = raw_err.decode(enc)
    return rc, output.strip(), err.strip()


def get_pytorch_env():
    return torch.utils.collect_env.get_env_info()


def track_current_environment() -> Environment:
    pytorch_info = get_pytorch_env()
    python_platform_info = get_python_platform_info()
    python_version = pytorch_info.python_version
    pytorch_version = pytorch_info.torch_version
    processor_info = python_platform_info[PROCESSOR]
    gpu_types = pytorch_info.nvidia_gpu_models
    pip_freeze = get_python_libs()

    return Environment(python_version=python_version, pytorch_version=pytorch_version, processor_info=processor_info,
                       gpu_types=gpu_types, pytorch_info=pytorch_info, python_platform_info=python_platform_info,
                       pip_freeze=pip_freeze)


if __name__ == '__main__':
    get_pytorch_env()

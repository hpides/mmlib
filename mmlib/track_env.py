import locale
import platform
import subprocess
import warnings

import torch
from torch.utils.collect_env import get_pretty_env_info

from mmlib.schema.environment import Environment

ARCHITECTURE = 'architecture'
MACHINE = 'machine'
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
MAC_VER = 'mac_ver'
LIBC_VER = 'libc_ver'


def get_python_platform_info():
    python_env_dict = {
        ARCHITECTURE: platform.architecture(),
        MACHINE: platform.machine(),
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
    print('get_pytorch_env -- takes relatively long...')
    pytorch_info = get_pytorch_env()
    python_platform_info = get_python_platform_info()
    python_version = pytorch_info.python_version
    pytorch_version = pytorch_info.torch_version
    processor_info = python_platform_info[PROCESSOR]
    gpu_types = pytorch_info.nvidia_gpu_models
    pip_freeze = get_python_libs()

    return Environment(python_version=python_version, pytorch_version=pytorch_version, processor_info=processor_info,
                       gpu_types=str(gpu_types), pytorch_info=str(pytorch_info),
                       python_platform_info=str(python_platform_info), pip_freeze=pip_freeze)


def compare_env_to_current(to_compare: Environment) -> bool:
    current_env = track_current_environment()

    if not current_env.python_version == to_compare.python_version:
        warnings.warn('Environment: The python version differs')
        return False
    if not current_env.pytorch_version == to_compare.pytorch_version:
        warnings.warn('Environment: The pytorch version differs')
        return False
    if not current_env.processor_info == to_compare.processor_info:
        warnings.warn('Environment: The processor info differs')
        return False
    if not current_env.gpu_types == to_compare.gpu_types:
        warnings.warn('Environment: The gpu types differ')
        return False
    if not current_env.pytorch_info == to_compare.pytorch_info:
        warnings.warn('Environment: The pytorch info differs')
        return False
    if not current_env.pip_freeze == to_compare.pip_freeze:
        warnings.warn('Environment: The installed python packages differ')
        return False
    if not current_env.python_platform_info == to_compare.python_platform_info:
        warnings.warn('Environment: The python platform info differs')
        return False

    return True


if __name__ == '__main__':
    get_pytorch_env()

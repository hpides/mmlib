import platform


def get_python_platform_info():
    python_env_dict = {
        'architecture': platform.architecture(),
        'machine': platform.machine(),
        'node': platform.node(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_build': platform.python_build(),
        'python_compiler': platform.python_compiler(),
        'python_branch': platform.python_branch(),
        'python_implementation': platform.python_implementation(),
        'python revision': platform.python_revision(),
        'python_version': platform.python_version(),
        'python_version_tuple': platform.python_version_tuple(),
        'release': platform.release(),
        'system': platform.system(),
        'version': platform.version(),
        'uname': platform.uname(),
        'mac_ver': platform.mac_ver(),
        'libc_ver': platform.libc_ver()
    }

    return python_env_dict

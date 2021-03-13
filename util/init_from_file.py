import os
import sys


def create_object(code, class_name):
    module, path = _module(code)
    sys.path.append(path)
    exec('from {} import {}'.format(module, class_name))
    model = eval('{}()'.format(class_name))

    return model


def create_object_with_parameters(class_name: [str], init_args: dict, code: str = None, import_cmd: str = None,
                                  init_ref_type_args: dict = None):
    if code:
        module, path = _module(code)
        sys.path.append(path)
        exec('from {} import {}'.format(module, class_name))
    else:
        exec(import_cmd)

    args_string = _arg_string(init_args)
    ref_args_type_strings = ''

    ref_type_refs = []
    if init_ref_type_args:
        ref_type_names = init_ref_type_args.keys()
        ref_type_refs = list(init_ref_type_args.values())
        for i, name in enumerate(ref_type_names):
            ref_args_type_strings += '{}=ref_type_refs[{}], '.format(name, i)

    template_string = '{}({})'
    if ref_args_type_strings:
        template_string = '{}({},{})'

    exec_str = template_string.format(class_name, args_string, ref_args_type_strings)
    model = eval(exec_str)

    return model


def _arg_string(init_args):
    args = []
    for k, v in init_args.items():
        if isinstance(v, str):
            args.append('{}=\'{}\''.format(k, v))
        else:
            args.append('{}={}'.format(k, v))

    return ', '.join(args)


def _module(code):
    path, file = os.path.split(code)
    module = file.replace('.py', '')
    return module, path

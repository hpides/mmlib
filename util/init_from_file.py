import os
import sys


def create_object(code, class_name):
    path, file = os.path.split(code)
    module = file.replace('.py', '')
    sys.path.append(path)
    exec('from {} import {}'.format(module, class_name))
    model = eval('{}()'.format(class_name))

    return model

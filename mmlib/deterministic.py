import torch

SEED = 42


def deterministic(func, f_args=None, f_kwargs=None):
    if f_kwargs is None:
        f_kwargs = {}
    if f_args is None:
        f_args = []
    # TODO check for deterministic dataloading:
    # https://discuss.pytorch.org/t/dataloader-is-not-deterministic/19250
    # WARNING, setting shuffle to False, drastically decreased model performance

    # TODO check if data loaders are deterministic
    # TODO maybe print warning for multiGPU

    set_deterministic()
    return func(*f_args, **f_kwargs)


def set_deterministic():
    # TODO maybe in the future we also have to set seed for used libraries, e.g. numpy
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    # TODO check if we can solve this more nicely
    # TODO check what to do for deterministic on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

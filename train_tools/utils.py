import torch
import numpy as np
import os, json, random

__all__ = ["ConfLoader", "directory_setter", "random_seeder"]


class ConfLoader:
    """
    Load json config file using DictWithAttributeAccess object_hook.
    ConfLoader(conf_name).opt attribute is the result of loading json config file.
    """

    class DictWithAttributeAccess(dict):
        """
        This inner class makes dict to be accessed same as class attribute.
        For example, you can use opt.key instead of the opt['key'].
        """

        def __getattr__(self, key):
            return self[key]

        def __setattr__(self, key, value):
            self[key] = value

    def __init__(self, conf_name):
        self.conf_name = conf_name
        self.opt = self.__get_opt()

    def __load_conf(self):
        with open(self.conf_name, "r") as conf:
            opt = json.load(
                conf, object_hook=lambda dict: self.DictWithAttributeAccess(dict)
            )
        return opt

    def __get_opt(self):
        opt = self.__load_conf()
        opt = self.DictWithAttributeAccess(opt)

        return opt


def directory_setter(path="./results", make_dir=False):
    """Create a directory when requested and validate that the path is a directory."""
    if not os.path.exists(path) and make_dir:
        os.makedirs(path)  # Create the directory if it does not exist.
        print("directory %s is created" % path)

    if not os.path.isdir(path):
        raise NotADirectoryError(
            "%s is not valid. set make_dir=True to make dir." % path
        )


def random_seeder(seed):
    """Fix random states used by Python, NumPy, PyTorch, CUDA, and MONAI."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        from monai.utils import set_determinism

        set_determinism(seed=seed)
    except Exception:
        pass

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except TypeError:
        torch.use_deterministic_algorithms(True)

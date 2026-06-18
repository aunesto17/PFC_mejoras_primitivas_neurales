import torch
import functools


class Optimizer(torch.optim.Optimizer):
    """Base optimizer for ESGD/ESGD_Max.
    Inherits from torch.optim.Optimizer for compatibility with PyTorch LR schedulers.
    """

    def __init__(self, params, defaults):
        # Ensure params is a list (not a generator) before passing to parent
        if not isinstance(params, (list, tuple)):
            params = list(params)
        super().__init__(params, defaults)

    def _hook_for_profile(self):
        self._zero_grad_profile_name = "Optimizer.zero_grad#{}.zero_grad".format(self.__class__.__name__)

        def profile_hook_step(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                obj, *_ = args
                profile_name = "Optimizer.step#{}.step".format(obj.__class__.__name__)
                with torch.autograd.profiler.record_function(profile_name):
                    return func(*args, **kwargs)
            return wrapper
        hooked = getattr(self.__class__.step, "hooked", None)
        if not hooked:
            self.__class__.step = profile_hook_step(self.__class__.step)
            self.__class__.step.hooked = True

# Lazy import functions to avoid optional module deps at import time

def train_tfa(*args, **kwargs):
    from .tfa import train_tfa_fast as _impl
    return _impl(*args, **kwargs)


def train_kochi_ppo(*args, **kwargs):
    from .kochi_ppo import train_kochi_ppo as _impl
    return _impl(*args, **kwargs)

__all__ = [
    "train_tfa",
    "train_kochi_ppo",
]

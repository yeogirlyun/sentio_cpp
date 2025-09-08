# Lazy import functions to avoid optional module deps at import time

def train_tfa_fast(*args, **kwargs):
    from .tfa_fast import train_tfa_fast as _impl
    return _impl(*args, **kwargs)


def train_tfa_seq(*args, **kwargs):
    from .tfa_seq import train_tfa_transformer as _impl
    return _impl(*args, **kwargs)


def train_kochi_ppo(*args, **kwargs):
    from .kochi_ppo import train_kochi_ppo as _impl
    return _impl(*args, **kwargs)

__all__ = [
    "train_tfa_fast",
    "train_tfa_seq",
    "train_kochi_ppo",
]

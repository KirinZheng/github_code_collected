from copy import deepcopy

from ..registry import MODELS

NODES = MODELS
SURROGATE = MODELS


# def check_init_shape(cfg):
#     if cfg['type'].startswith('State'):
#         return cfg
#     if 'init_state_shape' in cfg.keys():
#         cfg.pop('init_state_shape')
#     return cfg


def build_node(cfg):
    _cfg = deepcopy(cfg)
    # _cfg = check_init_shape(_cfg)
    if 'surrogate_function' in _cfg.keys():
        surrogate = build_surrogate(_cfg['surrogate_function'])
        _cfg['surrogate_function'] = surrogate
    return NODES.build(_cfg)


def build_surrogate(cfg):
    return SURROGATE.build(cfg)

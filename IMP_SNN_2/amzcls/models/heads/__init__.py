from .spike_cls_head import SpikeClsHead, TETClsHead, LogClsHead, MovingAverageHead, FixClsHead
from .spike_linear_head import SpikeLinearClsHead, TALinearClsHead, TETLinearClsHead

from .spike_linear_head_legacy import SpikeLinearClsHeadLegacy, TETLinearClsHeadLegacy, TALinearClsHeadLegacy

__all__ = [
    'SpikeClsHead', 'TETClsHead', 'LogClsHead', 'MovingAverageHead', 'FixClsHead',
    'SpikeLinearClsHead', 'TALinearClsHead', 'TETLinearClsHead',
    'SpikeLinearClsHeadLegacy', 'TETLinearClsHeadLegacy', 'TALinearClsHeadLegacy'
]

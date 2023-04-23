"""For Tensor Serialization"""
from __future__ import absolute_import

from .. import backend as F
from .._ffi.function import _init_api

__all__ = ["save_vc_map", "load_vc_map" ]

_init_api("dgl.data.vc_map_serialize")

def save_vc_map(filename, vc_map):
    assert False

def load_vc_map(filename, vc_map):
    assert False

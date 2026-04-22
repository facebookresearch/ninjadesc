# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict


def fix_state_dict_keys(net_key: str, state_dict: OrderedDict) -> OrderedDict:
    fixed_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if k.startswith(net_key):
            k_new = k[len(net_key) + 1 :]
            fixed_state_dict[k_new] = v

    return fixed_state_dict

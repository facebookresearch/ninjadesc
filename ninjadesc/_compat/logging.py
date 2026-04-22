# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging


def sudo_make_me_a_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

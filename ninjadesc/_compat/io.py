# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os


class _PathManager:
    def open(self, path, mode="r"):
        return open(path, mode)

    def exists(self, path):
        return os.path.exists(path)

    def mkdirs(self, path):
        os.makedirs(path, exist_ok=True)

    def get_local_path(self, path):
        return path


path_manager = _PathManager()


def clean_path(path):
    return os.path.normpath(path)

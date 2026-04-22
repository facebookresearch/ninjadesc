# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional

import hydra
import torchvision.transforms as T
from omegaconf import DictConfig


def instantiate_hydra_or_python(x: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(x, (DictConfig, dict)) and "_target_" in x:
        return hydra.utils.instantiate(x, *args, **kwargs)
    if callable(x):
        return x(*args, **kwargs)
    return x


def maybe_instantiate(cfg: Optional[DictConfig], key: str) -> Any:
    if cfg is None or key not in cfg or cfg[key] is None:
        return None
    return hydra.utils.instantiate(cfg[key])


def maybe_instantiate_list(cfg: Optional[DictConfig], key: str) -> List[Any]:
    if cfg is None or key not in cfg or cfg[key] is None:
        return []
    return [hydra.utils.instantiate(x) for x in cfg[key]]


def compose_transforms_from_hydra(cfg: Optional[DictConfig]) -> T.Compose:
    if cfg is None:
        return T.Compose([])
    return T.Compose([hydra.utils.instantiate(t) for t in cfg])

# Copied from ZFTurbo/Music-Source-Separation-Training (MIT, see LICENSE in this
# folder), models/bandit_v2, commit 1f700c3. Changes: plain nn.Module instead of a
# Lightning module, librosa formulas inlined. Original model: Watcharasupat et al.,
# https://github.com/kwatcharasupat/bandit-v2

from .bandit import Bandit

__all__ = ["Bandit"]

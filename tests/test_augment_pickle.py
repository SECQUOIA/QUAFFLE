"""DihedralAugmentDataset must survive pickling (Ray sends client datasets to workers)."""
import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.training_common import DihedralAugmentDataset, maybe_wrap_augment  # noqa: E402


class _Toy:
    def __init__(self, n=12):
        g = torch.Generator().manual_seed(0)
        self.items = [(torch.rand(3, 8, 8, generator=g), (torch.rand(1, 8, 8, generator=g) > 0.5).float())
                      for _ in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _draw(ds, n=10):
    return [ds[i % len(ds)] for i in range(n)]


def test_pickle_roundtrip_keeps_sequence():
    a = DihedralAugmentDataset(_Toy(), seed=17)
    b = pickle.loads(pickle.dumps(a))
    for (ia, ma), (ib, mb) in zip(_draw(a), _draw(b)):
        assert torch.equal(ia, ib) and torch.equal(ma, mb)


def test_pickle_mid_stream_keeps_sequence():
    a = DihedralAugmentDataset(_Toy(), seed=17)
    _draw(a, 3)
    b = pickle.loads(pickle.dumps(a))
    for (ia, ma), (ib, mb) in zip(_draw(a), _draw(b)):
        assert torch.equal(ia, ib) and torch.equal(ma, mb)


def test_reseed_after_unpickle():
    a = DihedralAugmentDataset(_Toy(), seed=17)
    b = pickle.loads(pickle.dumps(a)).reseed(17)
    c = DihedralAugmentDataset(_Toy(), seed=17)
    for (ib, mb), (ic, mc) in zip(_draw(b), _draw(c)):
        assert torch.equal(ib, ic) and torch.equal(mb, mc)


def test_maybe_wrap_augment_result_is_picklable():
    ds = maybe_wrap_augment(_Toy(), {'augment': True, 'seed': 314})
    assert isinstance(ds, DihedralAugmentDataset)
    pickle.loads(pickle.dumps(ds))


if __name__ == '__main__':
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print('PASS', name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print('FAIL', name, repr(e))
    sys.exit(1 if fails else 0)

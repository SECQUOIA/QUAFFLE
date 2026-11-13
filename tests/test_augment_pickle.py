"""DihedralAugmentDataset must survive pickling (Ray sends client datasets to workers)."""
import copyreg
import os
import pickle
import sys
from unittest import mock

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
    _draw(a, 3)
    b = pickle.loads(pickle.dumps(a)).reseed(23)
    c = DihedralAugmentDataset(_Toy(), seed=23)
    assert torch.equal(b.generator.get_state(), c.generator.get_state())
    for (ib, mb), (ic, mc) in zip(_draw(b), _draw(c)):
        assert torch.equal(ib, ic) and torch.equal(mb, mc)


def test_maybe_wrap_augment_result_is_picklable():
    ds = maybe_wrap_augment(_Toy(), {'augment': True, 'seed': 314})
    assert isinstance(ds, DihedralAugmentDataset)
    pickle.loads(pickle.dumps(ds))


def _reject_live_generator(generator):
    raise TypeError('A live torch.Generator reached the serializer')


def test_roundtrip_does_not_require_generator_pickling():
    import cloudpickle

    # Newer Torch versions can pickle generators themselves. Reject that path
    # explicitly so this guards the compatibility fix on those runtimes too.
    for serializer in (pickle, cloudpickle):
        a = DihedralAugmentDataset(_Toy(), seed=17)
        _draw(a, 3)
        before = a.generator.get_state().clone()
        with mock.patch.dict(copyreg.dispatch_table, {torch.Generator: _reject_live_generator}):
            payload = serializer.dumps(a)
            b = serializer.loads(payload)
        assert torch.equal(a.generator.get_state(), before)
        assert torch.equal(b.generator.get_state(), before)
        assert a.generator is not b.generator
        for (ia, ma), (ib, mb) in zip(_draw(a), _draw(b)):
            assert torch.equal(ia, ib) and torch.equal(ma, mb)


def test_cloudpickle_client_closure_keeps_independent_continuations():
    import cloudpickle
    from torch.utils.data import Subset

    datasets = [maybe_wrap_augment(Subset(_Toy(), [0, 2, 4]), {'augment': True, 'seed': 314})]
    _draw(datasets[0], 3)

    def client_fn(client_id):
        return datasets[int(client_id)]

    before = datasets[0].generator.get_state().clone()
    with mock.patch.dict(copyreg.dispatch_table, {torch.Generator: _reject_live_generator}):
        payload = cloudpickle.dumps(client_fn)
        restored = [cloudpickle.loads(payload)('0') for _ in range(2)]
    assert torch.equal(datasets[0].generator.get_state(), before)
    assert restored[0].generator is not restored[1].generator
    expected = _draw(datasets[0])
    for dataset in restored:
        assert torch.equal(dataset.generator.get_state(), before)
        for (ia, ma), (ib, mb) in zip(expected, _draw(dataset)):
            assert torch.equal(ia, ib) and torch.equal(ma, mb)


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

import numpy as np

from t0_training.olmo.cli import _iter_decoded_docs


class _FakeTokenizer:
    eos_token_id = 99

    @staticmethod
    def decode(tokens):
        return ",".join(str(token) for token in tokens)


def test_zero_eos_is_one_document():
    arr = np.array([1, 2, 3], dtype=np.uint32)

    assert list(_iter_decoded_docs(arr, _FakeTokenizer())) == ["1,2,3"]


def test_eos_separates_documents():
    arr = np.array([1, 2, 99, 3, 99], dtype=np.uint32)

    assert list(_iter_decoded_docs(arr, _FakeTokenizer())) == ["1,2", "3"]

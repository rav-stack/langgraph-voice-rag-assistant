import json
from eval.run import DATASET_PATH


def test_dataset_has_enough_examples():
    with open(DATASET_PATH) as f:
        data = json.load(f)

    assert len(data) >= 10                  #at least 10 test usecases
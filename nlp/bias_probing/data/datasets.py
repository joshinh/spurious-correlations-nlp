import os
from dataclasses import field, dataclass
from os.path import join
from typing import List, Tuple, Any, Callable

from datasets import load_dataset, DatasetDict
from torch import Tensor

_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class DatasetInfo:
    dataset_config_file: str = None
    eval_datasets: List[str] = field(default_factory=lambda: [])
    ignored_keys: List[str] = field(default_factory=lambda: [])
    sentence1_key: str = 'premise'
    sentence2_key: str = 'hypothesis'
    train_dataset_name: str = 'train'
    dev_dataset_name: str = 'validation'
    test_dataset_name: str = 'test'
    binerize: bool = False
    paired: bool = False
    mapper: Callable[[Any], Any] = None


def _fever_to_fever_symmetric_mapper(preds: Tensor) -> Tensor:
    # "SUPPORTS", "NOT ENOUGH INFO", "REFUTES" => "SUPPORTS", "REFUTES", "NOT ENOUGH INFO"
    preds[:, [1, 2]] = preds[:, [2, 1]]
    return preds


DATASETS = {
    'multi_nli': DatasetInfo(
        dataset_config_file='multi_nli.py',
        eval_datasets=['hans', 'hans::lexical_overlap', 'hans::subsequence', 'hans::constituent'],
        ignored_keys=['hypothesis_parse', 'premise_parse', 'id'],
        dev_dataset_name='validation_matched',
        test_dataset_name='validation_mismatched'
    ),
    'multi_nli_balanced': DatasetInfo(
        dataset_config_file='multi_nli_balanced.py',
        eval_datasets=['hans'],
        ignored_keys=['hypothesis_parse', 'premise_parse', 'id'],
        dev_dataset_name='validation_matched',
        test_dataset_name='validation_mismatched'
    ),
    'multi_nli_synthetic': DatasetInfo(
        dataset_config_file='multi_nli_synthetic.py',
        ignored_keys=['hypothesis_parse', 'premise_parse', 'id'],
        dev_dataset_name='validation_matched',
        test_dataset_name='validation_mismatched'
    ),
    'multi_nli_balanced_neg': DatasetInfo(
        dataset_config_file='multi_nli_balanced_neg.py',
        eval_datasets=['hans'],
        ignored_keys=['hypothesis_parse', 'premise_parse', 'id'],
        dev_dataset_name='validation_matched',
        test_dataset_name='validation_mismatched'
    ),
    'hans': DatasetInfo(
        test_dataset_name='validation',
        binerize=True,
        ignored_keys=['binary_parse_hypothesis', 'binary_parse_premise', 'heuristic',
                      'parse_hypothesis', 'parse_premise', 'subcase', 'template']
    ),
    'cad': DatasetInfo(
        dataset_config_file='cad.py',
        binerize=True
    )
}


def load_dataset_aux(train_dataset_name: str) -> Tuple[DatasetDict, DatasetInfo]:
    if train_dataset_name in DATASETS:
        dataset_info: DatasetInfo = DATASETS[train_dataset_name]
        train_dataset_name = os.path.abspath(join(_DIR, dataset_info.dataset_config_file)) \
            if dataset_info.dataset_config_file is not None else train_dataset_name
    else:
        dataset_info = DatasetInfo()
    datasets = load_dataset(train_dataset_name)
    return datasets, dataset_info

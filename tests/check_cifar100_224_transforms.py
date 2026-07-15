"""Verify identical Split CIFAR-100/224 transforms for prompt methods."""

from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys
import types

import yaml
from torchvision import transforms
from torchvision.transforms import InterpolationMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / 'datasets' / 'configs' / 'seq-cifar100-224'
MODEL_CONFIG_DIR = ROOT / 'models' / 'config'
METHOD_MODEL_CONFIGS = {
    'l2p': 'l2p.yaml',
    'dualprompt': 'dualprompt.yaml',
    'coda_prompt': 'coda_prompt.yaml',
}


# Avoid datasets/__init__.py autodiscovery: it imports every dataset, including
# seq_8vision and its optional OpenCLIP dependency. Submodules used by the real
# SequentialCIFAR100224 class remain importable through this namespace package.
sys.path.insert(0, str(ROOT))
datasets_pkg = types.ModuleType('datasets')
datasets_pkg.__path__ = [str(ROOT / 'datasets')]
sys.modules['datasets'] = datasets_pkg

from datasets.seq_cifar100_224 import SequentialCIFAR100224


def dataset_args():
    return Namespace(
        joint=False,
        custom_task_order=None,
        custom_class_order=None,
        permute_classes=False,
        label_perc=1,
        label_perc_by_class=1,
        seed=0,
    )


def resolve_dataset_config(model_config_name):
    with open(MODEL_CONFIG_DIR / model_config_name, encoding='utf-8') as config_file:
        model_config = yaml.safe_load(config_file)
    return model_config['seq-cifar100-224']['dataset_config']


def instantiate_dataset(method, model_config_name):
    dataset_config = resolve_dataset_config(model_config_name)
    with open(CONFIG_DIR / f'{dataset_config}.yaml', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    # Use a separate subclass because set_default_from_config intentionally
    # stores resolved transforms as class attributes.
    dataset_class = type(
        f'{method.title().replace("_", "")}CIFAR100224',
        (SequentialCIFAR100224,),
        {},
    )
    dataset_class.set_default_from_config(config, ArgumentParser(add_help=False))
    return dataset_config, dataset_class(dataset_args())


def assert_resize(resize):
    assert isinstance(resize, transforms.Resize)
    assert tuple(resize.size) == (224, 224), resize
    assert resize.interpolation == InterpolationMode.BICUBIC, resize
    assert resize.antialias is True, resize


def assert_train_transform(transform):
    assert [type(step) for step in transform.transforms] == [
        transforms.Resize,
        transforms.RandomHorizontalFlip,
        transforms.ToTensor,
        transforms.Normalize,
    ]
    assert_resize(transform.transforms[0])
    assert transform.transforms[1].p == 0.5
    assert transform.transforms[3].mean == [0, 0, 0]
    assert transform.transforms[3].std == [1, 1, 1]


def assert_test_transform(transform):
    assert [type(step) for step in transform.transforms] == [
        transforms.Resize,
        transforms.ToTensor,
        transforms.Normalize,
    ]
    assert_resize(transform.transforms[0])
    assert transform.transforms[2].mean == [0, 0, 0]
    assert transform.transforms[2].std == [1, 1, 1]


resolved_datasets = {
    method: instantiate_dataset(method, model_config_name)
    for method, model_config_name in METHOD_MODEL_CONFIGS.items()
}
datasets = {method: resolved[1] for method, resolved in resolved_datasets.items()}

for method, dataset in datasets.items():
    print(f'{method} (dataset_config={resolved_datasets[method][0]}):')
    print('  train:')
    print(dataset.TRANSFORM)
    print('  test:')
    print(dataset.TEST_TRANSFORM)
    assert_train_transform(dataset.TRANSFORM)
    assert_test_transform(dataset.TEST_TRANSFORM)

reference = datasets['l2p']
for method, dataset in datasets.items():
    assert repr(dataset.TRANSFORM) == repr(reference.TRANSFORM), f'{method} train differs'
    assert repr(dataset.TEST_TRANSFORM) == repr(reference.TEST_TRANSFORM), f'{method} test differs'

assert 'datasets.seq_8vision' not in sys.modules
assert 'open_clip' not in sys.modules
print('train transforms: identical')
print('test transforms: identical')
print('seq_8vision/OpenCLIP imported: no')
print('OK')

"""Check that prompt methods load the same pretrained ViT-B/16 base weights."""

from argparse import Namespace
from pathlib import Path
import sys
import types

import timm
import torch
import torch.nn as nn


EXPECTED_TIMM_VERSION = '0.9.8'
MODEL_NAME = 'vit_base_patch16_224.augreg_in21k_ft_in1k'
ROOT = Path(__file__).resolve().parents[1]

if timm.__version__ != EXPECTED_TIMM_VERSION:
    raise RuntimeError(
        f'This check requires timm=={EXPECTED_TIMM_VERSION}, found {timm.__version__}. '
        f'Install it with: python -m pip install timm=={EXPECTED_TIMM_VERSION}'
    )

# Running a file from tests/ does not automatically put the repository root on
# sys.path. Stub only the top-level models package so importing the three
# implementations does not execute models/__init__.py, discover datasets, or
# initialize the OpenCLIP model imported by seq_8vision.
sys.path.insert(0, str(ROOT))
models_pkg = types.ModuleType('models')
models_pkg.__path__ = [str(ROOT / 'models')]
sys.modules['models'] = models_pkg

from models.coda_prompt_utils.model import Model as CodaPromptModel
from models.dualprompt_utils.vision_transformer import vit_base_patch16_224_dualprompt
from models.l2p_utils.vit_prompt import vit_base_patch16_224_l2p


load_results = {}
active_load = None
original_load_state_dict = nn.Module.load_state_dict


def traced_load_state_dict(self, state_dict, strict=True, *args, **kwargs):
    incompatible = original_load_state_dict(self, state_dict, strict=strict, *args, **kwargs)
    if active_load is not None:
        load_results.setdefault(active_load, []).append(incompatible)
    return incompatible


nn.Module.load_state_dict = traced_load_state_dict
try:
    active_load = 'l2p'
    l2p = vit_base_patch16_224_l2p(
        pretrained=True,
        num_classes=0,
        prompt_length=5,
        embedding_key='cls',
        prompt_init='uniform',
        prompt_pool=True,
        prompt_key=True,
        pool_size=10,
        top_k=5,
        batchwise_prompt=False,
        prompt_key_init='uniform',
        head_type='prompt',
        use_prompt_mask=False,
    )

    active_load = 'dualprompt'
    dualprompt = vit_base_patch16_224_dualprompt(
        pretrained=True,
        num_classes=0,
        prompt_length=5,
        embedding_key='cls',
        prompt_init='uniform',
        prompt_pool=True,
        prompt_key=True,
        pool_size=10,
        top_k=1,
        batchwise_prompt=True,
        prompt_key_init='uniform',
        head_type='token',
        use_prompt_mask=True,
        use_g_prompt=True,
        g_prompt_length=5,
        g_prompt_layer_idx=[0, 1],
        use_prefix_tune_for_g_prompt=True,
        use_e_prompt=True,
        e_prompt_layer_idx=[2, 3, 4],
        use_prefix_tune_for_e_prompt=True,
        same_key_value=False,
        args=Namespace(use_permute_fix=False),
    )

    active_load = 'coda_prompt'
    coda_prompt = CodaPromptModel(
        num_classes=10,
        pt=True,
        prompt_param=[10, [100, 8, 0]],
    )
finally:
    active_load = None
    nn.Module.load_state_dict = original_load_state_dict


def print_load_result(name, result):
    print(f'{name}:')
    print(f'  missing_keys={result.missing_keys}')
    print(f'  unexpected_keys={result.unexpected_keys}')


assert len(load_results.get('l2p', [])) == 1, load_results.get('l2p')
assert len(load_results.get('dualprompt', [])) == 1, load_results.get('dualprompt')
# CODA-Prompt first loads the source checkpoint model, then loads that state
# dict into its prompt-aware ViT. The second result is the method load.
assert len(load_results.get('coda_prompt', [])) == 2, load_results.get('coda_prompt')

method_results = {
    'l2p': load_results['l2p'][0],
    'dualprompt': load_results['dualprompt'][0],
    'coda_prompt': load_results['coda_prompt'][-1],
}
allowed_missing_prefixes = {
    'l2p': ('prompt.',),
    'dualprompt': ('g_prompt', 'e_prompt.'),
    'coda_prompt': ('head.',),
}

print(f'checkpoint={MODEL_NAME}')
for method, result in method_results.items():
    print_load_result(method, result)
    assert not result.unexpected_keys, (method, result.unexpected_keys)
    assert all(key.startswith(allowed_missing_prefixes[method]) for key in result.missing_keys), (method, result.missing_keys)


@torch.no_grad()
def assert_module_equal(name, expected, *actuals):
    expected_state = expected.state_dict()
    for actual in actuals:
        actual_state = actual.state_dict()
        assert expected_state.keys() == actual_state.keys(), name
        for key in expected_state:
            assert torch.equal(expected_state[key], actual_state[key]), f'{name}.{key}'
    print(f'{name}: identical')


coda_vit = coda_prompt.feat
assert_module_equal('patch_embed', coda_vit.patch_embed, l2p.patch_embed, dualprompt.patch_embed)
assert_module_equal('blocks', coda_vit.blocks, l2p.blocks, dualprompt.blocks)
assert_module_equal('norm', coda_vit.norm, l2p.norm, dualprompt.norm)

assert torch.equal(coda_vit.cls_token, l2p.cls_token)
assert torch.equal(coda_vit.cls_token, dualprompt.cls_token)
print('cls_token: identical')

# L2P inserts prompt positional slots between the cls slot and the patch grid.
num_patches = coda_vit.patch_embed.num_patches
l2p_base_pos_embed = torch.cat((l2p.pos_embed[:, :1], l2p.pos_embed[:, -num_patches:]), dim=1)
assert torch.equal(coda_vit.pos_embed, l2p_base_pos_embed)
assert torch.equal(coda_vit.pos_embed, dualprompt.pos_embed)
print('base pos_embed: identical')
print('OK')

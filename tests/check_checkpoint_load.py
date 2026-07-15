"""Check that prompt methods load the same pretrained ViT-B/16 base weights."""

from argparse import Namespace
import gc
import hashlib
from pathlib import Path
import sys
import types

import timm
import torch


MODEL_NAME = 'vit_base_patch16_224.augreg_in21k_ft_in1k'
LEGACY_MODEL_NAME = 'vit_base_patch16_224'
AUGREG_URL = (
    'https://storage.googleapis.com/vit_models/augreg/'
    'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--'
    'imagenet2012-steps_20k-lr_0.01-res_224.npz'
)
ROOT = Path(__file__).resolve().parents[1]


def resolve_reference_model_name():
    available_models = set(timm.list_models(pretrained=True))
    if MODEL_NAME in available_models:
        return MODEL_NAME
    if LEGACY_MODEL_NAME in available_models:
        return LEGACY_MODEL_NAME
    raise RuntimeError(
        f'Neither {MODEL_NAME} nor its timm 0.4.12 alias {LEGACY_MODEL_NAME} '
        f'is available in timm=={timm.__version__}.'
    )


def state_digest(state_dict):
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        digest.update(name.encode())
        digest.update(state_dict[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


reference_model_name = resolve_reference_model_name()
reference = timm.create_model(reference_model_name, pretrained=True, num_classes=0)
reference_cfg = getattr(reference, 'pretrained_cfg', getattr(reference, 'default_cfg', {}))
checkpoint_url = reference_cfg.get('url')
if checkpoint_url != AUGREG_URL:
    raise RuntimeError(
        f'{reference_model_name} in timm=={timm.__version__} resolves to {checkpoint_url!r}, '
        f'not the expected AugReg artifact {AUGREG_URL!r}.'
    )

reference_state = reference.state_dict()
checkpoint_keys = set(reference_state)
checkpoint_digest = state_digest(reference_state)
reference_digests = {
    'patch_embed': state_digest(reference.patch_embed.state_dict()),
    'blocks': state_digest(reference.blocks.state_dict()),
    'norm': state_digest(reference.norm.state_dict()),
    'cls_token': state_digest({'cls_token': reference.cls_token}),
    'pos_embed': state_digest({'pos_embed': reference.pos_embed}),
}
del reference_state
del reference
gc.collect()

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

coda_prompt = CodaPromptModel(
    num_classes=10,
    pt=True,
    prompt_param=[10, [100, 8, 0]],
)
coda_vit = coda_prompt.feat


def load_result(model):
    # timm 0.4.12's custom NPZ loader copies tensors directly and does not
    # return _IncompatibleKeys, so compute the equivalent key report against
    # the resolved checkpoint. Exact tensor equality is asserted below.
    model_keys = set(model.state_dict())
    return sorted(model_keys - checkpoint_keys), sorted(checkpoint_keys - model_keys)


method_results = {
    'l2p': load_result(l2p),
    'dualprompt': load_result(dualprompt),
    'coda_prompt': load_result(coda_vit),
}
allowed_missing_prefixes = {
    'l2p': ('prompt.',),
    'dualprompt': ('g_prompt', 'e_prompt.'),
    'coda_prompt': ('head.',),
}

print(f'timm_version={timm.__version__}')
print(f'checkpoint={MODEL_NAME}')
print(f'resolved_identifier={reference_model_name}')
print(f'checkpoint_url={checkpoint_url}')
print(f'checkpoint_sha256={checkpoint_digest}')
for method, (missing_keys, unexpected_keys) in method_results.items():
    print(f'{method}:')
    print(f'  missing_keys={missing_keys}')
    print(f'  unexpected_keys={unexpected_keys}')
    assert not unexpected_keys, (method, unexpected_keys)
    assert all(key.startswith(allowed_missing_prefixes[method]) for key in missing_keys), (method, missing_keys)


@torch.no_grad()
def assert_module_equal(name, expected, *actuals):
    expected_state = expected.state_dict()
    assert state_digest(expected_state) == reference_digests[name], f'{name} differs from the checkpoint'
    for actual in actuals:
        actual_state = actual.state_dict()
        assert expected_state.keys() == actual_state.keys(), name
        for key in expected_state:
            assert torch.equal(expected_state[key], actual_state[key]), f'{name}.{key}'
    print(f'{name}: identical')


assert_module_equal('patch_embed', coda_vit.patch_embed, l2p.patch_embed, dualprompt.patch_embed)
assert_module_equal('blocks', coda_vit.blocks, l2p.blocks, dualprompt.blocks)
assert_module_equal('norm', coda_vit.norm, l2p.norm, dualprompt.norm)

assert state_digest({'cls_token': coda_vit.cls_token}) == reference_digests['cls_token']
assert torch.equal(coda_vit.cls_token, l2p.cls_token)
assert torch.equal(coda_vit.cls_token, dualprompt.cls_token)
print('cls_token: identical')

# L2P inserts prompt positional slots between the cls slot and the patch grid.
num_patches = coda_vit.patch_embed.num_patches
l2p_base_pos_embed = torch.cat((l2p.pos_embed[:, :1], l2p.pos_embed[:, -num_patches:]), dim=1)
assert state_digest({'pos_embed': coda_vit.pos_embed}) == reference_digests['pos_embed']
assert torch.equal(coda_vit.pos_embed, l2p_base_pos_embed)
assert torch.equal(coda_vit.pos_embed, dualprompt.pos_embed)
print('base pos_embed: identical')
print('OK')


import pytest
import torch
import torch.nn.functional as F

from fla.modules.l2norm import l2_norm
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 1, 60, torch.float),
            (2, 500, 4, 64, torch.float),
            (2, 1000, 2, 100, torch.float),
            (3, 1024, 4, 128, torch.float),
            (4, 1024, 5, 1024, torch.float16),
            (4, 1024, 5, 1024, torch.bfloat16),
            (5, 1024, 6, 2048, torch.float16),
            (5, 1024, 6, 2048, torch.bfloat16),
        ]
    ],
)
def test_l2norm(B: int, T: int, H: int, D: int, dtype: torch.dtype):
    torch.manual_seed(42)
    x = torch.randn(B, T, H, D, dtype=dtype).to(device).requires_grad_(True)
    x = x * 0.5 + 0.3

    ref = F.normalize(x, dim=-1, p=2)
    tri = l2_norm(x)
    ref_dx = torch.autograd.grad(ref.sum(), x)[0]
    tri_dx = torch.autograd.grad(tri.sum(), x)[0]

    assert_close('y', ref, tri, 0.005)
    assert_close('dx', ref_dx, tri_dx, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-scale{}".format(*test))
        for test in [
            # small D (uses l2norm_fwd_kernel, D <= 512)
            (2, 512, 4, 64, 1.0),
            (2, 512, 4, 64, 10.0),
            (2, 512, 4, 64, 100.0),
            (3, 1024, 4, 128, 1.0),
            (3, 1024, 4, 128, 10.0),
            (3, 1024, 4, 128, 100.0),
            # large D (uses l2norm_fwd_kernel1, D > 512)
            (2, 512, 4, 1024, 1.0),
            (2, 512, 4, 1024, 10.0),
            (2, 512, 4, 1024, 100.0),
            (3, 256, 4, 2048, 1.0),
            (3, 256, 4, 2048, 10.0),
            (3, 256, 4, 2048, 100.0),
        ]
    ],
)
def test_l2norm_bf16_stability(B: int, T: int, H: int, D: int, scale: float):
    """Test that l2norm produces stable (no NaN/Inf) outputs in bf16.

    Large input magnitudes stress bf16 range limits during the norm
    reduction. This catches instabilities in the rstd computation that
    can propagate through downstream layers like chunk gated delta rule.
    """
    torch.manual_seed(42)
    x = (torch.randn(B, T, H, D, dtype=torch.bfloat16).to(device) * scale).requires_grad_(True)

    tri = l2_norm(x)

    # Forward: no NaN/Inf, and unit-norm rows
    assert not torch.isnan(tri).any(), f"NaN in l2norm forward output (scale={scale})"
    assert not torch.isinf(tri).any(), f"Inf in l2norm forward output (scale={scale})"
    row_norms = tri.float().norm(dim=-1)
    assert_close('unit_norm', row_norms, torch.ones_like(row_norms), 0.01)

    # Compare against float32 reference
    ref = F.normalize(x.float(), dim=-1, p=2).bfloat16()
    assert_close('y_vs_f32ref', ref, tri, 0.01)

    # Backward: no NaN/Inf
    loss = tri.sum()
    tri_dx = torch.autograd.grad(loss, x)[0]
    assert not torch.isnan(tri_dx).any(), f"NaN in l2norm backward output (scale={scale})"
    assert not torch.isinf(tri_dx).any(), f"Inf in l2norm backward output (scale={scale})"

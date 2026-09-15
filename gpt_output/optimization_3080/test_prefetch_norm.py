"""Compare fused prediction norms with the checkpoint's BF16 cast boundaries."""
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from utils.prefetch_norm import fused_prefetch_norms


def check_norms():
    generator = torch.Generator(device='cuda').manual_seed(927)
    def random():
        return torch.randn((1, 2048), device='cuda', dtype=torch.bfloat16, generator=generator)
    def norm(x, layer):
        xf = x.float()
        normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + layer.variance_epsilon)
        return layer.weight * normalized.to(x.dtype)
    for use_residual in (False, True):
        raw, shared = random(), random()
        residual = random().reshape(1, 1, 2048) if use_residual else None
        n1 = SimpleNamespace(weight=(1 + random().reshape(-1) / 8), variance_epsilon=1e-6)
        n2 = SimpleNamespace(weight=(1 + random().reshape(-1) / 8), variance_epsilon=1e-6)
        h = raw + shared
        if residual is not None:
            h = h + residual.reshape_as(h)
        expected = norm(norm(h, n1) + h, n2)
        actual = fused_prefetch_norms(raw, shared, residual, n1, n2)
        torch.testing.assert_close(actual, expected, rtol=1/64, atol=1/128)
        print('[prefetch norm test]', use_residual, 'max_abs_error=',
              (actual.float() - expected.float()).abs().max().item(), flush=True)
        assert fused_prefetch_norms(raw.repeat(2, 1), shared.repeat(2, 1), None, n1, n2) is None


if __name__ == '__main__':
    check_norms()

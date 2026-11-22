import pytest
import torch
import torch.nn as nn

from gpn.star import model as star_model

_LEGACY_ERROR = (
    "PyTorch is checking whether allow_tf32_new is enabled for cuBlas matmul, "
    "Current status indicate that you have used mix of the legacy and new APIs "
    "to set the TF32 status for cublas matmul."
)


class _NewApiOnlyMatmul:
    """Stands in for torch.backends.cuda.matmul once TF32 was set the new way."""

    def __init__(self, precision):
        self.fp32_precision = precision

    @property
    def allow_tf32(self):
        raise RuntimeError(_LEGACY_ERROR)

    @allow_tf32.setter
    def allow_tf32(self, value):
        raise AssertionError("the legacy flag must not be written")


@pytest.mark.parametrize("precision", ["tf32", "ieee", "none"])
def test_tf32_context_never_touches_the_legacy_flag(monkeypatch, precision):
    fake = _NewApiOnlyMatmul(precision)
    monkeypatch.setattr(star_model, "_MATMUL", fake)
    monkeypatch.setattr(star_model, "_NEW_TF32_API", True)
    with star_model._tf32_matmul():
        assert fake.fp32_precision == "tf32"
    assert fake.fp32_precision == precision


def test_tf32_context_restores_the_setting_after_an_error(monkeypatch):
    fake = _NewApiOnlyMatmul("ieee")
    monkeypatch.setattr(star_model, "_MATMUL", fake)
    monkeypatch.setattr(star_model, "_NEW_TF32_API", True)
    with pytest.raises(ValueError):
        with star_model._tf32_matmul():
            raise ValueError
    assert fake.fp32_precision == "ieee"


class _LegacyOnlyMatmul:
    def __init__(self, allow):
        self.allow_tf32 = allow


@pytest.mark.parametrize("allow", [True, False])
def test_tf32_context_on_a_torch_without_the_new_api(monkeypatch, allow):
    fake = _LegacyOnlyMatmul(allow)
    monkeypatch.setattr(star_model, "_MATMUL", fake)
    monkeypatch.setattr(star_model, "_NEW_TF32_API", False)
    with star_model._tf32_matmul():
        assert fake.allow_tf32 is True
    assert fake.allow_tf32 is allow


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(
    not hasattr(torch.backends.cuda.matmul, "fp32_precision"),
    reason="torch has no fp32_precision",
)
@pytest.mark.parametrize("setter", ["none", "new_tf32", "new_ieee", "legacy"])
def test_compensated_linear_after_the_user_set_tf32(setter):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("compensated GEMM needs compute capability 8")
    matmul = torch.backends.cuda.matmul
    saved = matmul.fp32_precision
    try:
        if setter == "new_tf32":
            matmul.fp32_precision = "tf32"
        elif setter == "new_ieee":
            matmul.fp32_precision = "ieee"
        elif setter == "legacy":
            matmul.allow_tf32 = True
        before = matmul.fp32_precision
        torch.manual_seed(0)
        linear = nn.Linear(64, 48).cuda()
        x = torch.randn(32, 64, device="cuda")
        with torch.no_grad():
            expected = linear.double()(x.double()).float()
            linear = linear.float()
            out = star_model.compensated_linear(linear, x)
            cat = star_model.compensated_linear_cat(nn.Module(), (linear, linear), x)
        assert cat is not None
        assert (out.double() - expected.double()).abs().max() < 1e-4
        torch.testing.assert_close(cat[0], out, rtol=0, atol=1e-5)
        assert matmul.fp32_precision == before
    finally:
        matmul.fp32_precision = saved

from __future__ import annotations

import platform

import pytest
import torch

try:
    from open_provence.modeling_open_provence_standalone import _select_default_torch_dtype
except ImportError:  # datasets などが未インストールの場合はスキップ
    pytest.skip(
        "modeling_open_provence_standalone requires optional dependencies",
        allow_module_level=True,
    )


def test_select_default_dtype_cuda_prefers_bf16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert _select_default_torch_dtype("cuda") == torch.bfloat16


def test_select_default_dtype_cuda_fallback_float16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert _select_default_torch_dtype("cuda") == torch.float16


def test_select_default_dtype_cpu_apple(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    assert _select_default_torch_dtype("cpu") == "auto"


def test_select_default_dtype_mps(monkeypatch):
    assert _select_default_torch_dtype("mps") == "auto"


def test_select_default_dtype_unknown_device(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    assert _select_default_torch_dtype("cpu") is None

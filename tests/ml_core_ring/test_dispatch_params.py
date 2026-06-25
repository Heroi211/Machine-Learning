"""Testes ml_core_ring.dispatch_params."""

from __future__ import annotations

from ml_core_ring.dispatch_params import flatten_dispatch_train_params


def test_flatten_nested_params():
    conf = {
        "domain": "recommendation",
        "user_id": 2,
        "params": {"train_models": ["torch_embedding"], "n_epochs": 1},
        "top_k": 5,
    }
    out = flatten_dispatch_train_params(conf)
    assert out["train_models"] == ["torch_embedding"]
    assert out["n_epochs"] == 1
    assert out["top_k"] == 5
    assert "domain" not in out
    assert "user_id" not in out


def test_flat_conf_unchanged():
    conf = {
        "domain": "recommendation",
        "train_models": ["torch_embedding"],
        "n_epochs": 1,
        "top_k": 5,
    }
    out = flatten_dispatch_train_params(conf)
    assert out["train_models"] == ["torch_embedding"]
    assert out["n_epochs"] == 1


def test_top_level_overrides_nested():
    conf = {
        "params": {"n_epochs": 10, "train_models": ["popularity", "nmf"]},
        "train_models": ["torch_embedding"],
        "n_epochs": 1,
    }
    out = flatten_dispatch_train_params(conf)
    assert out["train_models"] == ["torch_embedding"]
    assert out["n_epochs"] == 1

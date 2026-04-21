from __future__ import annotations

import typing

import numpy as np
import pytest
import torch
from lightning.pytorch.cli import LightningArgumentParser

from sa_preisach.models import EncoderDecoderPreisachNN
from sa_preisach.nn import (
    PreisachEncoder,
    PreisachGRUEncoder,
    PreisachLSTMEncoder,
    PreisachRNNEncoder,
    PreisachTransformerEncoder,
)


def _build_encoder(
    encoder_cls: type[PreisachEncoder],
) -> PreisachEncoder:
    if encoder_cls is PreisachTransformerEncoder:
        return encoder_cls(
            num_features=2,
            d_model=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
    return typing.cast(
        PreisachEncoder,
        encoder_cls(
            num_features=2,
            hidden_dim=8,
            num_layers=1,
            dropout=0.0,
        ),
    )


@pytest.fixture
def fake_triangle_mesh(monkeypatch: pytest.MonkeyPatch) -> None:
    def _mock_triangle_mesh(
        mesh_scale: float,
        mesh_density_function: typing.Callable[..., np.ndarray] | None = None,
    ) -> np.ndarray:
        del mesh_scale
        del mesh_density_function
        return np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )

    monkeypatch.setattr(
        "sa_preisach.models._encoder_decoder_preisach_nn.create_triangle_mesh",
        _mock_triangle_mesh,
    )


@pytest.mark.parametrize(
    "encoder_cls",
    [
        PreisachLSTMEncoder,
        PreisachGRUEncoder,
        PreisachRNNEncoder,
        PreisachTransformerEncoder,
    ],
)
def test_encoder_interface_forward(encoder_cls: type[PreisachEncoder]) -> None:
    encoder = _build_encoder(encoder_cls)
    sequence = torch.rand(2, 6, 2)
    mesh_features = torch.rand(2, 3, 3)
    sequence_mask = torch.ones(2, 6, dtype=torch.bool)

    out = encoder(
        sequence=sequence,
        mesh_features=mesh_features,
        sequence_mask=sequence_mask,
    )

    assert out.shape == (2, 3)


@pytest.mark.parametrize(
    "encoder_cls",
    [
        PreisachLSTMEncoder,
        PreisachGRUEncoder,
        PreisachRNNEncoder,
        PreisachTransformerEncoder,
    ],
)
def test_encoder_decoder_model_instantiation_and_forward(
    encoder_cls: type[PreisachEncoder],
    fake_triangle_mesh: None,
) -> None:
    del fake_triangle_mesh
    encoder = _build_encoder(encoder_cls)
    model = EncoderDecoderPreisachNN(
        mesh_scale=0.5,
        hidden_dim=8,
        num_layers=2,
        compile_model=False,
        mesh_perturbation_std=0.0,
        encoder=encoder,
    )

    encoder_input = torch.rand(2, 5, 2)
    decoder_input = torch.rand(2, 4, 1)
    encoder_mask = torch.ones(2, 5, dtype=torch.bool)

    b_out, density, m_out, initial_states, mesh_coords = model(
        encoder_input=encoder_input,
        decoder_input=decoder_input,
        encoder_mask=encoder_mask,
    )

    assert b_out.shape == (2, 4)
    assert density.shape == (2, 3)
    assert m_out.shape == (2, 4)
    assert initial_states.shape == (2, 3)
    assert mesh_coords.shape == (2, 3, 2)


def test_encoder_decoder_model_requires_encoder(fake_triangle_mesh: None) -> None:
    del fake_triangle_mesh
    with pytest.raises(TypeError):
        EncoderDecoderPreisachNN(
            **typing.cast(
                typing.Any,
                {
                    "mesh_scale": 0.5,
                    "hidden_dim": 8,
                    "num_layers": 2,
                    "compile_model": False,
                    "mesh_perturbation_std": 0.0,
                },
            )
        )


@pytest.mark.parametrize(
    ("encoder_class_path", "expected_encoder_cls"),
    [
        ("sa_preisach.nn.PreisachLSTMEncoder", PreisachLSTMEncoder),
        ("sa_preisach.nn.PreisachGRUEncoder", PreisachGRUEncoder),
        ("sa_preisach.nn.PreisachRNNEncoder", PreisachRNNEncoder),
        ("sa_preisach.nn.PreisachTransformerEncoder", PreisachTransformerEncoder),
    ],
)
def test_lightning_parser_instantiates_model_encoder(
    encoder_class_path: str,
    expected_encoder_cls: type[PreisachEncoder],
    fake_triangle_mesh: None,
) -> None:
    del fake_triangle_mesh
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(EncoderDecoderPreisachNN, "model")

    encoder_init_args: dict[str, int | float] = {
        "num_features": 2,
        "hidden_dim": 8,
        "num_layers": 1,
        "dropout": 0.0,
    }
    if expected_encoder_cls is PreisachTransformerEncoder:
        encoder_init_args = {
            "num_features": 2,
            "d_model": 8,
            "num_heads": 2,
            "num_layers": 1,
            "dropout": 0.0,
        }

    config = parser.parse_object({
        "model": {
            "mesh_scale": 0.5,
            "hidden_dim": 8,
            "num_layers": 2,
            "compile_model": False,
            "mesh_perturbation_std": 0.0,
            "encoder": {
                "class_path": encoder_class_path,
                "init_args": encoder_init_args,
            },
        }
    })
    instantiated = parser.instantiate_classes(config)
    model = typing.cast(EncoderDecoderPreisachNN, instantiated["model"])

    assert isinstance(model.model.encoder, expected_encoder_cls)


def test_lightning_parser_model_requires_encoder(fake_triangle_mesh: None) -> None:
    del fake_triangle_mesh
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(EncoderDecoderPreisachNN, "model")

    with pytest.raises(SystemExit):
        parser.parse_object({
            "model": {
                "mesh_scale": 0.5,
                "hidden_dim": 8,
                "num_layers": 2,
                "compile_model": False,
                "mesh_perturbation_std": 0.0,
            }
        })


def test_adaptive_loss_weights_initial_values() -> None:
    from sa_preisach.models._adaptive_loss_weights import AdaptiveLossWeights

    alw = AdaptiveLossWeights(aux_loss_weight=2.0, saturation_reg_weight=0.5)
    assert torch.isclose(alw.log_seq.exp(), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(alw.log_aux.exp(), torch.tensor(2.0), atol=1e-5)
    assert torch.isclose(alw.log_sat.exp(), torch.tensor(0.5), atol=1e-5)


def test_adaptive_loss_weights_parameters_are_nn_parameters() -> None:
    from sa_preisach.models._adaptive_loss_weights import AdaptiveLossWeights

    expected_param_count = 3
    alw = AdaptiveLossWeights(aux_loss_weight=1.0, saturation_reg_weight=1.0)
    params = list(alw.parameters())
    assert len(params) == expected_param_count
    for p in params:
        assert isinstance(p, torch.nn.Parameter)


def test_prior_leaf_by_key_populated_with_composite(fake_triangle_mesh: None) -> None:
    import warnings

    from sa_preisach.priors import (
        CompositeDensityPrior,
        DiagonalDensityPrior,
        SymmetryDensityPrior,
    )

    del fake_triangle_mesh
    encoder = _build_encoder(PreisachLSTMEncoder)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prior = CompositeDensityPrior(
            DiagonalDensityPrior(weight=1.0),
            SymmetryDensityPrior(weight=1.0),
        )
    model = EncoderDecoderPreisachNN(
        mesh_scale=0.5,
        hidden_dim=8,
        num_layers=2,
        compile_model=False,
        mesh_perturbation_std=0.0,
        encoder=encoder,
        density_prior=prior,
    )
    assert "diagonal" in model._prior_leaf_by_key  # noqa: SLF001
    assert "symmetry" in model._prior_leaf_by_key  # noqa: SLF001
    assert len(model._prior_leaves) == 2  # noqa: SLF001, PLR2004


def test_prior_leaf_by_key_empty_without_prior(fake_triangle_mesh: None) -> None:
    del fake_triangle_mesh
    encoder = _build_encoder(PreisachLSTMEncoder)
    model = EncoderDecoderPreisachNN(
        mesh_scale=0.5,
        hidden_dim=8,
        num_layers=2,
        compile_model=False,
        mesh_perturbation_std=0.0,
        encoder=encoder,
    )
    assert model._prior_leaf_by_key == {}  # noqa: SLF001
    assert model._prior_leaves == []  # noqa: SLF001

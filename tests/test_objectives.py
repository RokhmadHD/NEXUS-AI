import pytest
import torch

from nexus_ai.models import NexusHSR
from nexus_ai.objectives import (
    TotalLoss,
    assignment_entropy_loss,
    causal_language_modeling_loss,
    fused_latent_consistency_loss,
    latent_state_alignment_loss,
)


def make_output():
    torch.manual_seed(0)
    return {
        "logits": torch.randn(2, 10, 32),
        "latent_state": torch.randn(2, 10, 16),
        "encoded_states": torch.randn(2, 10, 16),
        "fused_states": torch.randn(2, 10, 16),
        "symbolic_assignments": torch.softmax(torch.randn(2, 10, 8), dim=-1),
    }


def make_labels():
    labels = torch.randint(0, 32, (2, 10))
    labels[:, -3:] = -100  # padding positions
    return labels


def test_language_loss_ignores_padding():
    logits, labels = torch.randn(2, 10, 32), make_labels()
    loss, counted = causal_language_modeling_loss(logits, labels)
    assert counted == 14
    assert torch.isfinite(loss)

    reference = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 32), labels.reshape(-1), ignore_index=-100
    )
    assert torch.allclose(loss, reference)


def test_language_loss_rejects_unsupervised_labels():
    labels = torch.full((2, 10), -100)
    with pytest.raises(ValueError):
        causal_language_modeling_loss(torch.randn(2, 10, 32), labels)


def test_language_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        causal_language_modeling_loss(torch.randn(2, 10, 32), torch.randint(0, 32, (2, 9)))


def test_state_alignment_loss_is_zero_for_identical_states():
    states = torch.randn(2, 10, 16)
    assert latent_state_alignment_loss(states, states).item() == pytest.approx(0.0)
    assert latent_state_alignment_loss(states, states * 2).item() > 0


def test_consistency_loss_bounds():
    states = torch.randn(2, 10, 16)
    loss = fused_latent_consistency_loss(states, states)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    anti = fused_latent_consistency_loss(states, -states)
    assert anti.item() == pytest.approx(2.0)


def test_symbolic_entropy_prefers_peaked_assignments():
    uniform = torch.full((1, 1, 8), 1 / 8)
    peaked = torch.zeros(1, 1, 8)
    peaked[..., 0] = 1.0
    assert assignment_entropy_loss(peaked) < assignment_entropy_loss(uniform)


def test_total_loss_combines_components():
    breakdown = TotalLoss().compute(make_output(), make_labels())
    assert breakdown.counted_tokens == 14
    assert torch.isfinite(breakdown.total)
    expected = (
        breakdown.language
        + 0.1 * breakdown.state
        + 0.1 * breakdown.consistency
        + 0.01 * breakdown.symbolic
    )
    assert torch.allclose(breakdown.total, expected)


def test_total_loss_with_zero_weights_equals_language_loss():
    loss = TotalLoss(state_weight=0.0, consistency_weight=0.0, symbolic_weight=0.0)
    breakdown = loss.compute(make_output(), make_labels())
    assert torch.allclose(breakdown.total, breakdown.language)


def test_total_loss_validates_weights():
    with pytest.raises(ValueError):
        TotalLoss(language_weight=0.0)
    with pytest.raises(ValueError):
        TotalLoss(state_weight=-1.0)


def test_total_loss_requires_internal_states():
    with pytest.raises(KeyError):
        TotalLoss().compute({"logits": torch.randn(2, 10, 32)}, make_labels())


def test_gradients_flow_through_total_loss():
    model = NexusHSR(vocab_size=32, context_length=16, embedding_dim=32,
                     num_heads=4, encoder_layers=1, decoder_layers=1,
                     ffn_dim=64, latent_iterations=2, symbolic_slots=4)
    labels = torch.randint(0, 32, (2, 8))
    breakdown = TotalLoss().compute(model(torch.randint(0, 32, (2, 8))), labels)
    breakdown.total.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )

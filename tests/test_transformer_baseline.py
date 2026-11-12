import torch

from nexus_ai.models import NexusHSR, TransformerBaseline


def make_baseline():
    return TransformerBaseline(
        vocab_size=256,
        context_length=32,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        ffn_dim=192,
        dropout=0.0,
    )


def test_baseline_forward_shape_and_finite_logits():
    model = make_baseline()
    input_ids = torch.randint(0, 256, (2, 16))
    logits = model(input_ids)

    assert logits.shape == (2, 16, 256)
    assert torch.isfinite(logits).all()


def test_baseline_backward_produces_finite_gradients():
    model = make_baseline()
    input_ids = torch.randint(0, 256, (2, 16))
    targets = torch.randint(0, 256, (2, 16))

    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 256), targets.reshape(-1)
    )
    loss.backward()

    assert torch.isfinite(loss)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_baseline_is_causal():
    torch.manual_seed(11)
    model = make_baseline().eval()
    first = torch.randint(0, 256, (1, 12))
    second = first.clone()
    second[:, 8:] = torch.randint(0, 256, (1, 4))

    with torch.no_grad():
        logits_a = model(first)[:, :8]
        logits_b = model(second)[:, :8]

    assert torch.allclose(logits_a, logits_b, atol=1e-5, rtol=1e-5)


def test_baseline_rejects_sequence_over_context():
    model = make_baseline()
    input_ids = torch.randint(0, 256, (1, 33))

    try:
        model(input_ids)
    except ValueError as exc:
        assert "context_length" in str(exc)
    else:
        raise AssertionError("Expected ValueError for overlong sequence")


def test_default_baseline_matches_nexus_parameter_budget():
    nexus = NexusHSR()
    baseline = TransformerBaseline()

    nexus_count = nexus.count_parameters()
    baseline_count = baseline.count_parameters()
    difference_percent = abs(nexus_count - baseline_count) / nexus_count * 100

    assert difference_percent <= 0.1

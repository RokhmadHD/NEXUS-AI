import torch

from nexus_ai.models import NexusHSR


def make_model():
    return NexusHSR(
        vocab_size=256,
        context_length=32,
        embedding_dim=64,
        num_heads=4,
        encoder_layers=2,
        decoder_layers=2,
        ffn_dim=192,
        latent_iterations=2,
        symbolic_slots=8,
        dropout=0.0,
    )


def test_forward_shapes_and_finite_logits():
    model = make_model()
    input_ids = torch.randint(0, 256, (2, 16))
    output = model(input_ids)

    assert output["logits"].shape == (2, 16, 256)
    assert output["latent_state"].shape == (2, 16, 64)
    assert output["symbolic_assignments"].shape == (2, 16, 8)
    assert torch.isfinite(output["logits"]).all()


def test_backward_produces_gradients():
    model = make_model()
    input_ids = torch.randint(0, 256, (2, 16))
    targets = torch.randint(0, 256, (2, 16))

    logits = model(input_ids)["logits"]
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
    loss.backward()

    assert torch.isfinite(loss)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_model_rejects_sequence_longer_than_context():
    model = make_model()
    input_ids = torch.randint(0, 256, (1, 33))

    try:
        model(input_ids)
    except ValueError as exc:
        assert "context_length" in str(exc)
    else:
        raise AssertionError("Expected ValueError for overlong sequence")


def test_causal_prefix_is_independent_of_future_tokens():
    torch.manual_seed(7)
    model = make_model().eval()
    first = torch.randint(0, 256, (1, 12))
    second = first.clone()
    second[:, 8:] = torch.randint(0, 256, (1, 4))

    with torch.no_grad():
        logits_a = model(first)["logits"][:, :8]
        logits_b = model(second)["logits"][:, :8]

    assert torch.allclose(logits_a, logits_b, atol=1e-5, rtol=1e-5)


def test_parameter_count_is_positive():
    model = make_model()
    assert model.count_parameters() > 0

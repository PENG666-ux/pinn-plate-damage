"""Check the actual trainable sets and the manuscript's regularization policy."""

import torch

import config as C


def _assert_tail_trainable(model):
    for parameter in list(model.net.weights[:-2]) + list(model.net.biases[:-2]):
        assert not parameter.requires_grad
    for parameter in list(model.net.weights[-2:]) + list(model.net.biases[-2:]):
        assert parameter.requires_grad


def test_depth_counts_hidden_layers(model_factory, monkeypatch):
    monkeypatch.setattr(C, "DEPTH", 5)
    model = model_factory()
    assert len(model.net.weights) == 6
    assert [tuple(weight.shape) for weight in model.net.weights] == [
        (3, 8), (8, 8), (8, 8), (8, 8), (8, 8), (8, 1)
    ]


def test_experimental_calibration_updates_only_network_tail(model_factory):
    model = model_factory()
    model.set_stage(1)
    assert all(parameter.requires_grad for parameter in model.net.parameters())
    model.set_finetune_mode()
    _assert_tail_trainable(model)
    assert not any(
        parameter.requires_grad
        for parameter in [model.raw_alpha, model.raw_x_i,
                          model.raw_y_i, model.raw_r_i]
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.net.named_parameters()
    }
    model.train(epochs=1, batch=12, stage=1)
    _assert_tail_trainable(model)
    changed = []
    for name, parameter in model.net.named_parameters():
        if not parameter.requires_grad:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        else:
            changed.append(not torch.equal(parameter, before[name]))
    assert any(changed)
    torch.testing.assert_close(model.alpha, torch.zeros_like(model.alpha))


def test_inverse_training_freezes_radius_then_releases_it(
    model_factory, monkeypatch
):
    monkeypatch.setattr(C, "S2_EARLY_PHASE", 1)
    monkeypatch.setattr(C, "ADAPTIVE_UPDATE_INTERVAL", 100_000)
    model = model_factory()
    model.init_damage_params(
        positions=[(0.4, 0.5)], radii_mm=[12.0], alphas=[0.7]
    )
    observations = []
    original_loss = model.compute_loss

    def record_loss(*args, **kwargs):
        _assert_tail_trainable(model)
        observations.append((
            model.raw_r_i.requires_grad,
            model.raw_r_i.detach().clone(),
            model.optimizer.param_groups[-1]["lr"],
        ))
        return original_loss(*args, **kwargs)

    monkeypatch.setattr(model, "compute_loss", record_loss)
    frozen_head = model.net.weights[0].detach().clone()
    model.train(epochs=3, batch=12, stage=2)
    assert [entry[0] for entry in observations] == [False, True, True]
    torch.testing.assert_close(
        observations[0][1], observations[1][1], rtol=0, atol=0
    )
    assert not torch.equal(observations[1][1], observations[2][1])
    assert observations[0][2] == C.LR_DAMAGE
    assert observations[1][2] == C.LR_DAMAGE_FINE
    torch.testing.assert_close(model.net.weights[0], frozen_head, rtol=0, atol=0)


def test_binary_gate_penalty_does_not_select_a_single_defect(
    model_factory, monkeypatch
):
    monkeypatch.setattr(C, "K_MAX", 3)
    model = model_factory()
    model.set_stage(3)
    model.init_damage_params(alphas=[0.99, 0.01, 0.01])
    one_active = model._sparsity_loss(0).detach()
    model.init_damage_params(alphas=[0.99, 0.99, 0.01])
    two_active = model._sparsity_loss(5000)
    torch.testing.assert_close(one_active, two_active)
    expected = model.w_reg_g * torch.sum(model.alpha * (1 - model.alpha))
    torch.testing.assert_close(two_active, expected)


def test_radius_regularizer_shrinks_inactive_candidates_without_target_radius(
    model_factory, monkeypatch
):
    monkeypatch.setattr(C, "K_MAX", 3)
    model = model_factory()
    model.set_stage(3)
    model.init_damage_params(
        radii_mm=[3.0, 7.0, 11.0], alphas=[0.9, 0.8, 0.1]
    )
    loss = model._radius_loss()
    expected = model.w_reg_r * torch.sum((1 - model.alpha) * model.r_i_norm**2)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert bool(torch.all(model.raw_r_i.grad > 0))


def test_lbfgs_line_search_reuses_one_objective(model_factory, monkeypatch):
    """Repeated Wolfe evaluations at unchanged weights must be identical."""
    values = []

    class RepeatedClosureOptimizer:
        def __init__(self, parameters, **kwargs):
            self.parameters = list(parameters)

        def zero_grad(self):
            for parameter in self.parameters:
                parameter.grad = None

        def step(self, closure):
            values.extend([closure().item(), closure().item()])

    model = model_factory()
    model.set_stage(1)
    monkeypatch.setattr(torch.optim, "LBFGS", RepeatedClosureOptimizer)
    model.train_lbfgs(epochs=1, subset=4, max_iter=2)
    assert len(values) == 2
    assert values[0] == values[1]

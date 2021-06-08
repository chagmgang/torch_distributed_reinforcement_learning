import collections

import torch
import torch.nn as nn
import torch.nn.functional as F


VTraceFromLogitsReturns = collections.namedtuple(
    "VTraceFromLogitsReturns",
    [
        "vs",
        "pg_advantages",
        "log_rhos",
        "behavior_action_log_probs",
        "target_action_log_probs",
    ],
)

VTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages")

def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())

def action_log_probs(policy_logits, actions):
    return -F.nll_loss(
        F.log_softmax(torch.flatten(policy_logits, 0, -2), dim=-1),
        torch.flatten(actions),
        reduction="none",
    ).view_as(actions)


def from_logits(
    behavior_policy_logits,
    target_policy_logits,
    actions,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    """V-trace for softmax policies."""

    target_action_log_probs = action_log_probs(target_policy_logits, actions)
    behavior_action_log_probs = action_log_probs(behavior_policy_logits, actions)
    log_rhos = target_action_log_probs - behavior_action_log_probs
    vtrace_returns = from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
    )
    return VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behavior_action_log_probs=behavior_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        **vtrace_returns._asdict(),
    )


@torch.no_grad()
def from_importance_weights(
    log_rhos,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    """V-trace from log importance weights."""
    with torch.no_grad():
        rhos = torch.exp(log_rhos)
        if clip_rho_threshold is not None:
            clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
        else:
            clipped_rhos = rhos

        cs = torch.clamp(rhos, max=1.0)
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        )
        deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

        acc = torch.zeros_like(bootstrap_value)
        result = []
        for t in range(discounts.shape[0] - 1, -1, -1):
            acc = deltas[t] + discounts[t] * cs[t] * acc
            result.append(acc)
        result.reverse()
        vs_minus_v_xs = torch.stack(result)

        # Add V(x_s) to get v_s.
        vs = torch.add(vs_minus_v_xs, values)

        # Advantage for policy gradient.
        broadcasted_bootstrap_values = torch.ones_like(vs[0]) * bootstrap_value
        vs_t_plus_1 = torch.cat(
            [vs[1:], broadcasted_bootstrap_values.unsqueeze(0)], dim=0
        )
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
        else:
            clipped_pg_rhos = rhos
        pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)

        # Make sure no gradients backpropagated through the returned values.
        return VTraceReturns(vs=vs, pg_advantages=pg_advantages)

class ImpalaLoss(nn.Module):

    def __init__(self, discount=0.99, baseline_coef=1.0, ent_coef=0.05):
        super(ImpalaLoss, self).__init__()

        self.discount = discount
        self.baseline_coef = baseline_coef
        self.ent_coef = ent_coef

    def forward(self, behavior_policy_logits,
                target_policy_logits, action, values,
                next_values, reward, done):

        discounts = (1 - done).float() * self.discount

        actor_outputs = torch.transpose(behavior_policy_logits, 0, 1)
        learner_outputs = torch.transpose(target_policy_logits, 0, 1)
        actions = torch.transpose(action, 0, 1)
        discounts = torch.transpose(discounts, 0, 1)
        rewards = torch.transpose(reward, 0, 1)
        values = torch.transpose(values, 0, 1)
        bootstrap_value = torch.transpose(next_values, 0, 1)[-1]

        vtrace_returns = from_logits(
                behavior_policy_logits=actor_outputs,
                target_policy_logits=learner_outputs,
                actions=actions,
                discounts=discounts,
                rewards=rewards,
                values=values,
                bootstrap_value=bootstrap_value)

        pg_loss = compute_policy_gradient_loss(
                learner_outputs,
                actions,
                vtrace_returns.pg_advantages)

        baseline_loss = compute_baseline_loss(
                vtrace_returns.vs - values)

        entropy_loss = compute_entropy_loss(
            learner_outputs
        )

        total_loss = pg_loss + baseline_loss * self.baseline_coef + entropy_loss * self.ent_coef

        return total_loss, pg_loss, baseline_loss, entropy_loss

if __name__ == '__main__':

    learner_outputs_policy_logits = torch.as_tensor([
        [[0.32610413, 0.0, 0.0],
         [0.34757757, 0.0, 0.0],
         [0.0, 0.0, 0.3514619],
         [0.0, 0.0, 0.3324681],
         [0.0, 0.32490963, 0.0]]])

    actor_outputs_policy_logits = torch.as_tensor([
        [[0.33032224, 0.0, 0.0],
         [0.34173246, 0.0, 0.0],
         [0.0, 0.0, 0.33941314],
         [0.0, 0.0, 0.32420245],
         [0.0, 0.3259509, 0.0]]])

    actions = torch.as_tensor([
        [0, 0, 2, 2, 1]]).to(torch.long)

    softmax = torch.nn.Softmax(dim=-1)
    learner_prob = softmax(learner_outputs_policy_logits)
    learner_log_softmax = torch.log(learner_prob)
    actor_prob = softmax(actor_outputs_policy_logits)
    actor_log_softmax = torch.log(actor_prob)

    print(learner_log_softmax)
    print(actor_log_softmax)

    learner_outputs_policy_logits = torch.transpose(learner_outputs_policy_logits, 0, 1)
    actions = torch.transpose(actions, 0, 1)

    print(action_log_probs(learner_outputs_policy_logits, actions))

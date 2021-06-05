import torch
import numpy as np

def split_data(x):
    return x[:, :-2], x[:, 1:-1], x[:, 2:]

def log_probs_from_softmax_and_actions(prob, actions):

    batch, traj, action_size = prob.shape
    onehot_action = torch.zeros_like(prob)
    for a, o in zip(actions, onehot_action):
        o[range(traj), a] = 1
    probs = torch.sum(prob * onehot_action, dim=2)
    log_probs = torch.log(probs + 1e-8)
    return log_probs

def from_softmax(behavior_policy, target_policy, discounts, actions,
                 rewards, values, next_values, clip_rho_threshold=1.0,
                 clip_pg_rho_threshold=1.0):

    target_log_probs = log_probs_from_softmax_and_actions(
            target_policy, actions)
    behavior_log_probs = log_probs_from_softmax_and_actions(
            behavior_policy, actions)

    log_rhos = target_log_probs - behavior_log_probs

    transpose_log_rhos = torch.transpose(log_rhos, 0, 1)
    transpose_discounts = torch.transpose(discounts, 0, 1)
    transpose_rewards = torch.transpose(rewards, 0, 1)
    transpose_values = torch.transpose(values, 0, 1)
    transpose_next_values = torch.transpose(next_values, 0, 1)

    vs, clipped_pg_rhos = from_importance_weights(
            transpose_log_rhos, transpose_discounts,
            transpose_rewards, transpose_values,
            bootstrap_value=transpose_next_values[-1])

    vs = torch.transpose(vs, 0, 1)
    clipped_pg_rhos = torch.transpose(clipped_pg_rhos, 0, 1)

    return vs.detach(), clipped_pg_rhos.detach()

def from_importance_weights(log_rhos, discounts, rewards,
                            values, bootstrap_value, clip_rho_threshold=1.0,
                            clip_pg_rho_threshold=1.0):

    rhos = torch.exp(log_rhos)
    if clip_rho_threshold is not None:
        clipped_rhos = torch.clamp_max(rhos, clip_rho_threshold)
    else:
        clipped_rhos = rhos

    cs = torch.clamp_max(rhos, 1.0)
    values_t_plus_1 = torch.cat(
        [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)
    deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

    vs_minus_v_xs = [torch.zeros_like(bootstrap_value)]
    for i in reversed(range(len(discounts))):
        discount_t, c_t, delta_t = discounts[i], cs[i], deltas[i]
        vs_minus_v_xs.append(delta_t + discount_t * c_t * vs_minus_v_xs[-1])
    vs_minus_v_xs = torch.stack(vs_minus_v_xs[1:])
    # Reverse the results back to original order.
    vs_minus_v_xs = torch.flip(vs_minus_v_xs, dims=[0])
    # Add V(x_s) to get v_s.
    vs = vs_minus_v_xs + values

    if clip_pg_rho_threshold is not None:
        clipped_pg_rhos = torch.clamp_max(rhos, clip_pg_rho_threshold)
    else:
        clipped_pg_rhos = rhos

    return vs.detach(), clipped_pg_rhos.detach()

def compute_policy_gradient_loss(pi, actions, advantage):
    log_prob = log_probs_from_softmax_and_actions(
            pi, actions)
    advantage = advantage.detach()
    policy_gradient_loss_per_timestep = log_prob * advantage
    return -torch.sum(policy_gradient_loss_per_timestep)

def compute_baseline_loss(vs, values):
    error = vs.detach() - values
    l2_loss = error ** 2
    return torch.sum(l2_loss)

if __name__ == '__main__':
    
    values = torch.as_tensor([
        [-0.01613846, 0.00063246, -0.01603599, -0.01379921, -0.02884073]])

    next_values = torch.as_tensor([
        [0.00063246, -0.01603599, -0.01379921, -0.02884073, 0.00417265]])

    rewards = torch.as_tensor([
        [1, 1, 0, 0, 0]])

    discounts = torch.as_tensor([
        [0.99, 0.99, 0.99, 0, 0.99]])

    actions = torch.as_tensor([
        [0, 0, 2, 2, 1]]).to(torch.long)

    pi = torch.as_tensor([
        [[0.32610413, 0.0, 0.0],
         [0.34757757, 0.0, 0.0],
         [0.0, 0.0, 0.3514619],
         [0.0, 0.0, 0.3324681],
         [0.0, 0.32490963, 0.0]]])

    mu = torch.as_tensor([
        [[0.33032224, 0.0, 0.0],
         [0.34173246, 0.0, 0.0],
         [0.0, 0.0, 0.33941314],
         [0.0, 0.0, 0.32420245],
         [0.0, 0.3259509, 0.0]]])

    from_softmax(
            behavior_policy=mu, target_policy=pi,
            actions=actions, discounts=discounts,
            rewards=rewards, values=values,
            next_values=next_values)

import torch
from botorch.utils.multi_objective import pareto
from evaluators.utils import get_all_metrics, get_all_metrics_docker, get_all_metrics_pygmo
import numpy as np
import itertools
# from polyleven import levenshtein
import matplotlib.pyplot as plt


def pareto_frontier(solutions, rewards, maximize=True):
    pareto_mask = pareto.is_non_dominated(torch.tensor(rewards).cpu() if maximize else -torch.tensor(rewards).cpu())
    if solutions is not None:
        if solutions.shape[0] == 1:
            pareto_front = solutions
        else:
            pareto_front = solutions[pareto_mask]
    else:
        pareto_front = None
    if rewards.shape[0] == 1:
        pareto_rewards = rewards
    else:
        pareto_rewards = rewards[pareto_mask]
    return pareto_front, pareto_rewards


# def mean_pairwise_distances(seqs):
#     dists = []
#     for pair in itertools.combinations(seqs, 2):
#         dists.append(levenshtein(*pair))
#     return np.mean(dists)


def generate_simplex(dims, n_per_dim):
    spaces = [np.linspace(0.0, 1.0, n_per_dim) for _ in range(dims)]
    return np.array([comb for comb in itertools.product(*spaces)
                     if np.allclose(sum(comb), 1.0)])


def thermometer(v, n_bins=50, vmin=0, vmax=32):
    bins = torch.linspace(vmin, vmax, n_bins)
    gap = bins[1] - bins[0]
    return (v[..., None] - bins.reshape((1,) * v.ndim + (-1,))).clamp(0, gap.item()) / gap


def plot_pareto(pareto_rewards, all_rewards, pareto_only=False, objective_names=None):
    if pareto_rewards.shape[-1] < 3:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if not pareto_only:
            ax.scatter(*np.hsplit(all_rewards, all_rewards.shape[-1]), color="grey", label="All Samples")
        ax.scatter(*np.hsplit(pareto_rewards, pareto_rewards.shape[-1]), color="red", label="Pareto Front")
        if objective_names is not None:
            ax.set_xlabel(objective_names[0])
            ax.set_ylabel(objective_names[1])
        else:
            ax.set_xlabel("Reward 1")
            ax.set_ylabel("Reward 2")
        ax.legend()
        return fig
    if pareto_rewards.shape[-1] == 3:
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Scatter3d(
            x=all_rewards[:, 0],
            y=all_rewards[:, 1],
            z=all_rewards[:, 2],
            mode='markers',
            marker_color="grey",
            name="All Samples"
        ),
        go.Scatter3d(
            x=pareto_rewards[:, 0],
            y=pareto_rewards[:, 1],
            z=pareto_rewards[:, 2],
            mode='markers',
            marker_color="red",
            name="Pareto Front"
        )])
        if objective_names is not None:
            fig.update_layout(
                scene = dict(
                    xaxis_title=objective_names[0],
                    yaxis_title=objective_names[1],
                    zaxis_title=objective_names[2],
                )
            )
        fig.update_traces(marker=dict(size=8),
                  selector=dict(mode='markers'))
        return fig


def get_pareto_fronts(states, rewards):
    if states is not None:
        states = np.array(states)
    rewards = np.array(rewards)
    if rewards.ndim == 1:
        rewards = np.expand_dims(rewards, 0)

    pareto_candidates, pareto_rewards = pareto_frontier(states, rewards, maximize=True)
    return pareto_candidates, pareto_rewards


def get_hypervolume(states, rewards, num_objectives):
    pareto_candidates, pareto_rewards = get_pareto_fronts(states, rewards)
    simplex = generate_simplex(num_objectives, n_per_dim=10)
    mo_metrics = get_all_metrics(pareto_rewards, ["hypervolume", "r2"], hv_ref=[0]*num_objectives, r2_prefs=simplex,
                                 num_obj=num_objectives)
    # obj_names = self.task_cfg.objectives if hasattr(self.task_cfg, "objectives") else [f"obj_{i}" for i in
    #                                                                                    range(self.obj_dim)]
    #
    # fig = plot_pareto(pareto_targets, all_rewards, pareto_only=False, objective_names=obj_names) if plot else None
    return mo_metrics["hypervolume"], mo_metrics["r2"]

def get_hypervolume_docker(states, rewards, num_objectives):
    pareto_candidates, pareto_rewards = get_pareto_fronts(states, rewards)
    simplex = generate_simplex(num_objectives, n_per_dim=10)
    mo_metrics = get_all_metrics_docker(pareto_rewards, ["hypervolume", "r2"], hv_ref=[0]*num_objectives, r2_prefs=simplex,
                                 num_obj=num_objectives)
    # obj_names = self.task_cfg.objectives if hasattr(self.task_cfg, "objectives") else [f"obj_{i}" for i in
    #                                                                                    range(self.obj_dim)]
    #
    # fig = plot_pareto(pareto_targets, all_rewards, pareto_only=False, objective_names=obj_names) if plot else None
    return mo_metrics["hypervolume"], mo_metrics["r2"]

def get_hypervolume_pygmo(states, rewards, num_objectives):
    pareto_candidates, pareto_rewards = get_pareto_fronts(states, rewards)
    simplex = generate_simplex(num_objectives, n_per_dim=10)
    mo_metrics = get_all_metrics_pygmo(pareto_rewards, ["hypervolume", "r2"], hv_ref=[0]*num_objectives, r2_prefs=simplex,
                                 num_obj=num_objectives)
    # obj_names = self.task_cfg.objectives if hasattr(self.task_cfg, "objectives") else [f"obj_{i}" for i in
    #                                                                                    range(self.obj_dim)]
    #
    # fig = plot_pareto(pareto_targets, all_rewards, pareto_only=False, objective_names=obj_names) if plot else None
    return mo_metrics["hypervolume"], mo_metrics["r2"]



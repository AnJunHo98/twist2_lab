"""MLP model with LayerNorm after each hidden layer."""

from __future__ import annotations

import torch.nn as nn
from functools import reduce
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, resolve_nn_activation


class MLPWithLayerNorm(nn.Sequential):
    """MLP with LayerNorm inserted after each hidden linear layer, before activation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "elu",
        last_activation: str | None = None,
    ) -> None:
        super().__init__()

        activation_mod = resolve_nn_activation(activation)
        last_activation_mod = resolve_nn_activation(last_activation) if last_activation is not None else None
        hidden_dims_processed = [input_dim if dim == -1 else dim for dim in hidden_dims]

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims_processed[0]))
        layers.append(nn.LayerNorm(hidden_dims_processed[0]))
        layers.append(activation_mod)

        for i in range(len(hidden_dims_processed) - 1):
            layers.append(nn.Linear(hidden_dims_processed[i], hidden_dims_processed[i + 1]))
            layers.append(nn.LayerNorm(hidden_dims_processed[i + 1]))
            layers.append(activation_mod)

        if isinstance(output_dim, int):
            layers.append(nn.Linear(hidden_dims_processed[-1], output_dim))
        else:
            total_out_dim = reduce(lambda x, y: x * y, output_dim)
            layers.append(nn.Linear(hidden_dims_processed[-1], total_out_dim))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))

        if last_activation_mod is not None:
            layers.append(last_activation_mod)

        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)


class MLPModelLN(MLPModel):
    """MLPModel variant that uses LayerNorm after each hidden layer.

    Set class_name="twist2_lab.rl.mlp_model_ln.MLPModelLN" in RslRlMLPModelCfg
    to use this model instead of the default MLPModel.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        # Replicate MLPModel.__init__ but replace MLP with MLPWithLayerNorm
        nn.Module.__init__(self)

        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = nn.Identity()

        if distribution_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore[assignment]
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            raw_input_dim = self.distribution.input_dim
            mlp_output_dim: int = raw_input_dim if isinstance(raw_input_dim, int) else raw_input_dim[0]
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        self.mlp = MLPWithLayerNorm(
            self._get_latent_dim(),
            mlp_output_dim,
            list(hidden_dims),
            str(activation),
        )

        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

from functools import partial

from torch.optim import AdamW


def layerwise_lr_decay(decay_rate, name_dict, n_layers, param_name):
    """
    Calculate layer-wise learning rate decay ratio.

    Args:
        decay_rate (float):
            The layer-wise decay ratio.
        name_dict (dict):
            Dictionary mapping parameter names to full parameter names.
            Use dict(model.named_parameters()) or {p: n for n, p in model.named_parameters()}
            to get name_dict.
        n_layers (int):
            Total number of layers in the transformer encoder.
        param_name (str or torch.nn.Parameter):
            The parameter name or parameter object to calculate decay ratio for.

    Returns:
        float: The learning rate ratio for this parameter.
    """
    ratio = 1.0

    # Ensure param_name is a string
    # If name_dict values are parameter tensors, then param_name should be the key (string)
    # If name_dict values are strings, then param_name can be looked up
    if isinstance(param_name, str):
        # param_name is already a string, use it directly or look it up
        # name_dict can be either {name: param} or {name: name_string}
        value = name_dict.get(param_name, param_name)
        # If the value is a string, use it; otherwise, use param_name itself
        static_name = value if isinstance(value, str) else param_name
    else:
        # param_name is a parameter object, shouldn't happen in our implementation
        static_name = str(param_name)

    # Ensure static_name is a string before checking substring
    if not isinstance(static_name, str):
        static_name = str(static_name)

    if "blocks." in static_name or "layers." in static_name:
        idx_1 = static_name.find("blocks.")
        idx_2 = static_name.find("layers.")
        assert any([x >= 0 for x in [idx_1, idx_2]]), (
            f"Cannot find blocks or layers in {static_name}"
        )
        idx = idx_1 if idx_1 >= 0 else idx_2

        layer = int(static_name[idx:].split(".")[1])
        ratio = decay_rate ** (n_layers - layer)

    elif (
        "cls_token" in static_name
        or "patch_embed" in static_name
        or "pos_embed" in static_name
    ):
        ratio = decay_rate ** (n_layers + 1)

    return ratio


class AdamWDL(AdamW):
    r"""
    The AdamWDL optimizer is implemented based on the AdamW Optimization with dynamic lr setting.
    Generally it's used for transformer model.

    We use "layerwise_lr_decay" as default dynamic lr setting method of AdamWDL.
    "Layer-wise decay" means exponentially decaying the learning rates of individual
    layers in a top-down manner. For example, suppose the 24-th layer uses a learning
    rate l, and the Layer-wise decay rate is α, then the learning rate of layer m
    is lα^(24-m). See more details on: https://arxiv.org/abs/1906.08237.

    .. math::
        & t = t + 1

        & moment\_1\_out = {\beta}_1 * moment\_1 + (1 - {\beta}_1) * grad

        & moment\_2\_out = {\beta}_2 * moment\_2 + (1 - {\beta}_2) * grad * grad

        & learning\_rate = learning\_rate * \frac{\sqrt{1 - {\beta}_2^t}}{1 - {\beta}_1^t}

        & param\_out = param - learning\_rate * (\frac{moment\_1}{\sqrt{moment\_2} + \epsilon} + \lambda * param)

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        layerwise_decay (float, optional): The layer-wise decay ratio. Defaults to 1.0.
        n_layers (int, optional): The total number of encoder layers. Defaults to 12.
        set_param_lr_func (function|None, optional): If it's not None, set_param_lr_func() will set the the parameter
            learning rate before it executes Adam Optimizer. Defaults to layerwise_lr_decay.
        name_dict (dict, optional): The keys of name_dict is parameter name while the value
            of name_dict is the parameter's full name in the model.
            Use dict(model.named_parameters()) to get name_dict.

    Examples:
        >>> import torch
        >>> from detrs.optimizer import AdamWDL
        >>>
        >>> def simple_lr_setting(decay_rate, name_dict, n_layers, param_name):
        >>>     ratio = 1.0
        >>>     static_name = name_dict.get(param_name, param_name)
        >>>     if "weight" in static_name:
        >>>         ratio = decay_rate**0.5
        >>>     return ratio
        >>>
        >>> linear = torch.nn.Linear(10, 10)
        >>>
        >>> name_dict = dict(linear.named_parameters())
        >>>
        >>> adamwdl = AdamWDL(
        >>>     linear.parameters(),
        >>>     lr=1e-4,
        >>>     set_param_lr_func=simple_lr_setting,
        >>>     layerwise_decay=0.8,
        >>>     name_dict=name_dict)
        >>>
        >>> inp = torch.rand(10, 10)
        >>> out = linear(inp)
        >>> loss = out.mean()
        >>>
        >>> loss.backward()
        >>> adamwdl.step()
        >>> adamwdl.zero_grad()
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        layerwise_decay=1.0,
        n_layers=12,
        set_param_lr_func=None,
        name_dict=None,
    ):
        if not isinstance(layerwise_decay, (float, int)):
            raise TypeError("layerwise_decay should be float or int.")

        self.layerwise_decay = layerwise_decay
        self.n_layers = n_layers
        self.set_param_lr_func = (
            partial(set_param_lr_func, layerwise_decay, name_dict, n_layers)
            if set_param_lr_func is not None
            else None
        )
        self.name_dict = name_dict if name_dict is not None else {}
        self.base_lr = lr

        # Convert params to list if it's a generator
        if not isinstance(params, list):
            params = list(params)

        # Check if params is already a list of parameter groups (dicts)
        is_param_groups = (
            len(params) > 0 and isinstance(params[0], dict) and "params" in params[0]
        )

        # If set_param_lr_func is provided, create parameter groups with custom learning rates
        if self.set_param_lr_func is not None:
            param_groups = []

            # Create reverse mapping: parameter object -> parameter name
            param_to_name = {}
            for name, p in self.name_dict.items():
                param_to_name[p] = name

            # Extract all parameters from either flat list or param groups
            if is_param_groups:
                # Extract parameters from existing parameter groups
                all_params = []
                group_settings = []  # Store original group settings
                for group in params:
                    group_params = group["params"]
                    if not isinstance(group_params, list):
                        group_params = list(group_params)
                    for param in group_params:
                        all_params.append(param)
                        # Store group settings for this param
                        group_settings.append(
                            {k: v for k, v in group.items() if k != "params"}
                        )
                params_to_process = all_params
            else:
                params_to_process = params
                group_settings = [{}] * len(params)

            # Create parameter groups with layer-wise learning rates
            for param, settings in zip(params_to_process, group_settings):
                # Find parameter name using the reverse mapping
                param_name = param_to_name.get(param, None)

                if param_name is None:
                    # If not found in name_dict, use default lr
                    ratio = 1.0
                else:
                    ratio = self.set_param_lr_func(param_name)

                # Merge original settings with new lr
                group_dict = {
                    "params": [param],
                    "lr": lr * ratio,
                    **settings,  # Include weight_decay and other settings
                }
                param_groups.append(group_dict)

            super(AdamWDL, self).__init__(
                param_groups,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
            )
        else:
            super(AdamWDL, self).__init__(
                params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
            )


def build_adamwdl(
    model,
    lr=1e-4,
    weight_decay=0.05,
    betas=(0.9, 0.999),
    layer_decay=0.65,
    num_layers=None,
    filter_bias_and_bn=True,
    skip_decay_names=None,
    set_param_lr_func="layerwise_lr_decay",
):
    """
    Build AdamWDL optimizer with layer-wise learning rate decay.

    Args:
        model (torch.nn.Module): The model to optimize.
        lr (float): Base learning rate. Default: 1e-4.
        weight_decay (float): Weight decay coefficient. Default: 0.05.
        betas (tuple): Coefficients for computing running averages. Default: (0.9, 0.999).
        layer_decay (float): Layer-wise decay rate. Default: 0.65.
        num_layers (int): Number of layers. Default: None.
        filter_bias_and_bn (bool): Whether to filter bias and batch norm parameters
            from weight decay. Default: True.
        skip_decay_names (list): List of parameter name patterns to skip decay. Default: None.
        set_param_lr_func (str or callable): Function to set parameter-wise learning rate.
            Default: 'layerwise_lr_decay'.

    Returns:
        AdamWDL: The optimizer instance.
    """
    decay_dict = None
    parameters = None

    if filter_bias_and_bn or skip_decay_names:
        # Create decay dictionary to control which parameters get weight decay
        decay_dict = {}
        for name, param in model.named_parameters():
            # Check if should skip decay
            should_decay = True

            if filter_bias_and_bn:
                # Skip decay for 1D parameters (biases, norms) and bias parameters
                if len(param.shape) == 1 or name.endswith(".bias"):
                    should_decay = False

            if skip_decay_names and should_decay:
                # Check against skip patterns
                if any([_n in name for _n in skip_decay_names]):
                    should_decay = False

            decay_dict[name] = should_decay

        # Create parameter groups based on decay_dict
        decay_params = [
            p for n, p in model.named_parameters() if decay_dict.get(n, True)
        ]
        no_decay_params = [
            p for n, p in model.named_parameters() if not decay_dict.get(n, True)
        ]

        param_groups = []
        if decay_params:
            param_groups.append({"params": decay_params, "weight_decay": weight_decay})
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

        parameters = param_groups if param_groups else model.parameters()
    else:
        parameters = model.parameters()

    # Prepare optimizer arguments
    opt_args = {
        "params": parameters,
        "lr": lr,
        "weight_decay": weight_decay
        if decay_dict is None
        else 0.0,  # Use 0.0 if already set in param_groups
        "betas": betas,
        "layerwise_decay": layer_decay,
    }

    # Set learning rate function
    if isinstance(set_param_lr_func, str):
        set_param_lr_func = eval(set_param_lr_func)

    if set_param_lr_func is not None:
        opt_args["set_param_lr_func"] = set_param_lr_func
        # Create name dictionary for parameter lookup
        name_dict = dict(model.named_parameters())
        opt_args["name_dict"] = name_dict
        opt_args["n_layers"] = num_layers

    optimizer = AdamWDL(**opt_args)

    return optimizer

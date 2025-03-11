r"""All In One Block Layer.

This module provides an invertible block that combines multiple flow operations:
affine coupling, permutation, and global affine transformation.

The block performs the following computation:

.. math::

    y = V R \; \Psi(s_\mathrm{global}) \odot \mathrm{Coupling}
    \Big(R^{-1} V^{-1} x\Big)+ t_\mathrm{global}

where:

- :math:`V` is an optional learned householder reflection matrix
- :math:`R` is a permutation matrix
- :math:`\Psi` is an activation function for global scaling
- The coupling operation splits input :math:`x` into :math:`x_1, x_2` and outputs
  :math:`u = \mathrm{concat}(u_1, u_2)` where:

  .. math::

      u_1 &= x_1 \odot \exp \Big( \alpha \; \mathrm{tanh}\big( s(x_2) \big)\Big)
      + t(x_2) \\
      u_2 &= x_2

Example:
    >>> import torch
    >>> from anomalib.models.components.flow import AllInOneBlock
    >>> # Create flow block
    >>> def subnet_fc(c_in, c_out):
    ...     return torch.nn.Sequential(
    ...         torch.nn.Linear(c_in, 128),
    ...         torch.nn.ReLU(),
    ...         torch.nn.Linear(128, c_out)
    ...     )
    >>> flow = AllInOneBlock(
    ...     dims_in=[(64,)],
    ...     subnet_constructor=subnet_fc
    ... )
    >>> # Apply flow transformation
    >>> x = torch.randn(10, 64)
    >>> y, logdet = flow(x)
    >>> print(y[0].shape)
    torch.Size([10, 64])

Args:
    dims_in (list[tuple[int]]): Dimensions of input tensor(s)
    dims_c (list[tuple[int]], optional): Dimensions of conditioning tensor(s).
        Defaults to None.
    subnet_constructor (Callable, optional): Function that constructs the subnet,
        called as ``f(channels_in, channels_out)``. Defaults to None.
    affine_clamping (float, optional): Clamping value for affine coupling.
        Defaults to 2.0.
    gin_block (bool, optional): Use GIN coupling from Sorrenson et al, 2019.
        Defaults to False.
    global_affine_init (float, optional): Initial value for global affine
        scaling. Defaults to 1.0.
    global_affine_type (str, optional): Type of activation for global affine
        scaling. One of ``'SIGMOID'``, ``'SOFTPLUS'``, ``'EXP'``.
        Defaults to ``'SOFTPLUS'``.
    permute_soft (bool, optional): Use soft permutation matrix from SO(N).
        Defaults to False.
    learned_householder_permutation (int, optional): Number of learned
        householder reflections. Defaults to 0.
    reverse_permutation (bool, optional): Apply inverse permutation before block.
        Defaults to False.

Raises:
    ValueError: If ``subnet_constructor`` is None or dimensions are invalid.
"""

# Copyright (c) https://github.com/vislearn/FrEIA
# SPDX-License-Identifier: MIT

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Callable
import math
from typing import Any

import torch
from FrEIA.modules import InvertibleModule
from scipy.stats import special_ortho_group
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torch.nn import init
from torch.distributions.utils import lazy_property

logger = logging.getLogger(__name__)


def _global_scale_sigmoid_activation(input_tensor: torch.Tensor) -> torch.Tensor:
    """Apply sigmoid activation for global scaling.

    Args:
        input_tensor (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Scaled tensor after sigmoid activation
    """
    return 10 * torch.sigmoid(input_tensor - 2.0)


def _global_scale_softplus_activation(input_tensor: torch.Tensor) -> torch.Tensor:
    """Apply softplus activation for global scaling.

    Args:
        input_tensor (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Scaled tensor after softplus activation
    """
    softplus = nn.Softplus(beta=0.5)
    return 0.1 * softplus(input_tensor)


def _global_scale_exp_activation(input_tensor: torch.Tensor) -> torch.Tensor:
    """Apply exponential activation for global scaling.

    Args:
        input_tensor (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Scaled tensor after exponential activation
    """
    return torch.exp(input_tensor)


class AllInOneBlock(InvertibleModule):
    r"""Module combining common operations in normalizing flows.

    It combines affine or additive coupling, permutation, and global affine transformation
    ('ActNorm'). It can also be used as GIN coupling block, perform learned
    householder permutations, and use an inverted pre-permutation. The affine
    transformation includes a soft clamping mechanism, first used in Real-NVP.
    The block as a whole performs the following computation:

    .. math::

        y = V R \; \Psi(s_\mathrm{global}) \odot \mathrm{Coupling}\Big(R^{-1} V^{-1} x\Big)+ t_\mathrm{global}

    - The inverse pre-permutation of x (i.e. :math:`R^{-1} V^{-1}`) is optional (see
      ``reverse_permutation`` below).
    - The learned householder reflection matrix
      :math:`V` is also optional all together (see ``learned_householder_permutation``
      below).
    - For the coupling, the input is split into :math:`x_1, x_2` along
      the channel dimension. Then the output of the coupling operation is the
      two halves :math:`u = \mathrm{concat}(u_1, u_2)`.

      .. math::

          u_1 &= x_1 \odot \exp \Big( \alpha \; \mathrm{tanh}\big( s(x_2) \big)\Big) + t(x_2) \\
          u_2 &= x_2

      Because :math:`\mathrm{tanh}(s) \in [-1, 1]`, this clamping mechanism prevents
      exploding values in the exponential. The hyperparameter :math:`\alpha` can be adjusted.

    Args:
        subnet_constructor: class or callable ``f``, called as ``f(channels_in, channels_out)`` and
            should return a torch.nn.Module. Predicts coupling coefficients :math:`s, t`.
        affine_clamping: clamp the output of the multiplicative coefficients before
            exponentiation to +/- ``affine_clamping`` (see :math:`\alpha` above).
        gin_block: Turn the block into a GIN block from Sorrenson et al, 2019.
            Makes it so that the coupling operations as a whole is volume preserving.
        global_affine_init: Initial value for the global affine scaling :math:`s_\mathrm{global}`.
        global_affine_init: ``'SIGMOID'``, ``'SOFTPLUS'``, or ``'EXP'``. Defines the activation to be used
            on the beta for the global affine scaling (:math:`\Psi` above).
        permute_soft: bool, whether to sample the permutation matrix :math:`R` from :math:`SO(N)`,
            or to use hard permutations instead. Note, ``permute_soft=True`` is very slow
            when working with >512 dimensions.
        learned_householder_permutation: Int, if >0, turn on the matrix :math:`V` above, that represents
            multiple learned householder reflections. Slow if large number.
            Dubious whether it actually helps network performance.
        reverse_permutation: Reverse the permutation before the block, as introduced by Putzky
            et al, 2019. Turns on the :math:`R^{-1} V^{-1}` pre-multiplication above.
        affine_coupling: If True, use affine coupling layers, otherwise use additive coupling layers.
    """

    def __init__(
        self,
        dims_in: list[tuple[int]],
        dims_c: list[tuple[int]] | None = None,
        subnet_constructor: Callable | None = None,
        affine_clamping: float = 2.0,
        gin_block: bool = False,
        global_affine_init: float = 1.0,
        global_affine_type: str = "SOFTPLUS",
        permute_soft: bool = False,
        learned_householder_permutation: int = 0,
        reverse_permutation: bool = True,
        # TODO: (Note) Added parameters
        permute: bool = True,
        use_prior: bool = False,
        # Change the following to use a uniformly scaling flow
        affine_coupling: bool = False,
        bijective_affine_transform: bool = True,
        reverse_bijective_affine_transform: bool = True,
    ) -> None:
        if dims_c is None:
            dims_c = []
        super().__init__(dims_in, dims_c)

        channels = dims_in[0][0]
        # rank of the tensors means 1d, 2d, 3d tensor etc.
        self.input_rank = len(dims_in[0]) - 1
        # tuple containing all dims except for batch-dim (used at various points)
        self.sum_dims = tuple(range(1, 2 + self.input_rank))

        if len(dims_c) == 0:
            self.conditional = False
            self.condition_channels = 0
        else:
            if tuple(dims_c[0][1:]) != tuple(dims_in[0][1:]):
                msg = f"Dimensions of input and condition don't agree: {dims_c} vs {dims_in}."
                raise ValueError(msg)

            self.conditional = True
            self.condition_channels = sum(dc[0] for dc in dims_c)

        split_len1 = channels - channels // 2
        split_len2 = channels // 2
        self.splits = [split_len1, split_len2]

        try:
            self.permute_function = {0: F.linear, 1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[self.input_rank]
        except KeyError:
            msg = f"Data is {1 + self.input_rank}D. Must be 1D-4D."
            raise ValueError(msg) from None

        self.in_channels = channels
        self.clamp = affine_clamping
        self.GIN = gin_block
        self.reverse_pre_permute = reverse_permutation
        self.householder = learned_householder_permutation

        if permute_soft and channels > 512:
            msg = (
                "Soft permutation will take a very long time to initialize "
                f"with {channels} feature channels. Consider using hard permutation instead."
            )
            logger.warning(msg)

        # global_scale is used as the initial value for the global affine scale
        # (pre-activation). It is computed such that
        # the 'magic numbers' (specifically for sigmoid) scale the activation to
        # a sensible range.
        if global_affine_type == "SIGMOID":
            global_scale = 2.0 - torch.log(torch.tensor([10.0 / global_affine_init - 1.0]))
            self.global_scale_activation = _global_scale_sigmoid_activation
        elif global_affine_type == "SOFTPLUS":
            global_scale = 2.0 * torch.log(torch.exp(torch.tensor(0.5 * 10.0 * global_affine_init)) - 1)
            self.global_scale_activation = _global_scale_softplus_activation
        elif global_affine_type == "EXP":
            global_scale = torch.log(torch.tensor(global_affine_init))
            self.global_scale_activation = _global_scale_exp_activation
        else:
            message = 'Global affine activation must be "SIGMOID", "SOFTPLUS" or "EXP"'
            raise ValueError(message)

        self.global_scale = nn.Parameter(torch.ones(1, self.in_channels, *([1] * self.input_rank)) * global_scale)
        self.global_offset = nn.Parameter(torch.zeros(1, self.in_channels, *([1] * self.input_rank)))

        self.permute = permute
        if permute_soft:
            w = special_ortho_group.rvs(channels)
        else:
            indices = torch.randperm(channels)
            w = torch.zeros((channels, channels))
            w[torch.arange(channels), indices] = 1.0

        if self.householder:
            # instead of just the permutation matrix w, the learned housholder
            # permutation keeps track of reflection vectors vk, in addition to a
            # random initial permutation w_0.
            self.vk_householder = nn.Parameter(0.2 * torch.randn(self.householder, channels), requires_grad=True)
            self.w_perm = None
            self.w_perm_inv = None
            self.w_0 = nn.Parameter(torch.FloatTensor(w), requires_grad=False)
        else:
            self.w_perm = nn.Parameter(
                torch.FloatTensor(w).view(channels, channels, *([1] * self.input_rank)),
                requires_grad=False,
            )
            self.w_perm_inv = nn.Parameter(
                torch.FloatTensor(w.T).view(channels, channels, *([1] * self.input_rank)),
                requires_grad=False,
            )
            
        # LU transform
        self.bijective_affine_transform = bijective_affine_transform
        self.reverse_bijective_affine_transform = \
            reverse_bijective_affine_transform
        self.use_LU = \
            bijective_affine_transform | reverse_bijective_affine_transform
        if self.use_LU:
            self.L_raw = torch.nn.Parameter(torch.empty(channels, channels)) 
            self.U_raw = torch.nn.Parameter(torch.empty(channels, channels)) 
            self.LU_bias = torch.nn.Parameter(torch.empty(channels)) 
            self.prior_scale = 1.0 # TODO: configurable?

            self.input_shape = channels

            # Triangular matrices
            self.L_mask = torch.tril(torch.ones(channels, channels), diagonal=-1)
            self.U_mask = torch.triu(torch.ones(channels, channels), diagonal=0)

            # Hooks to zero out off-diagonal gradients
            self.L_raw.register_hook(lambda grad: grad * self.L_mask.to(grad.device))
            # Add gradient prior corrector
            # (equivalent to adding a logprior term to the loss function)
            if use_prior:
                self.U_raw.register_hook(
                    lambda grad: grad * self.U_mask.to(grad.device) + \
                        self._logprior_grad_corrector()  
                )
            else:
                self.U_raw.register_hook(
                    lambda grad: grad * self.U_mask.to(grad.device) 
                )
            # Parameter initialization
            init.kaiming_uniform_(self.L_raw, nonlinearity="relu")
            with torch.no_grad():
                self.L_raw.copy_(self.L_raw.tril(diagonal=-1).fill_diagonal_(1))

            init.kaiming_uniform_(self.U_raw, nonlinearity="relu")
            
            with torch.no_grad():
                self.U_raw.fill_diagonal_(0) 
                #self.U_raw += torch.eye(self.channel)
                # TODO: Proper handling
                d = channels
                sign = -torch.ones(d) + 2 * torch.bernoulli(.5 * torch.ones(d))
                scale = self.prior_scale * torch.ones(d) * 1/d \
                    if self.prior_scale is not None else torch.ones(d) 
                
                self.U_raw += \
                    sign * torch.normal(torch.zeros(d), scale).exp().diag()
                self.U_raw.copy_(self.U_raw.triu())

            if self.LU_bias is not None:
                fan_in = channels
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.LU_bias, -bound, bound)
            

        if subnet_constructor is None:
            message = "Please supply a callable subnet_constructor function or object (see docstring)"
            raise ValueError(message)
        self.subnet = subnet_constructor(self.splits[0] + self.condition_channels, 2 * self.splits[1])
        self.last_jac = None
        
        # Coupling type
        self.affine_coupling = affine_coupling
    
    def to(self, device):
        super().to(device)
        self.L_mask = self.L_mask.to(device)
        self.R_mask = self.R_mask.to(device)

    def _construct_householder_permutation(self) -> torch.Tensor:
        """Compute permutation matrix from learned reflection vectors.

        Returns:
            torch.Tensor: Constructed permutation matrix
        """
        w = self.w_0
        for vk in self.vk_householder:
            w = torch.mm(w, torch.eye(self.in_channels).to(w.device) - 2 * torch.ger(vk, vk) / torch.dot(vk, vk))

        for _ in range(self.input_rank):
            w = w.unsqueeze(-1)
        return w

    def _permute(self, x: torch.Tensor, rev: bool = False) -> tuple[Any, float | torch.Tensor]:
        """Perform permutation and scaling after coupling operation.

        Args:
            x (torch.Tensor): Input tensor
            rev (bool, optional): Reverse the permutation. Defaults to False.

        Returns:
            tuple[Any, float | torch.Tensor]: Transformed outputs and LogJacDet
                of scaling
        """
        if self.GIN:
            scale = 1.0
            perm_log_jac = 0.0
        else:
            scale = self.global_scale_activation(self.global_scale)
            perm_log_jac = torch.sum(torch.log(scale))

        if rev:
            return ((self.permute_function(x, self.w_perm_inv) - self.global_offset) / scale, perm_log_jac)

        return (self.permute_function(x * scale + self.global_offset, self.w_perm), perm_log_jac)
    

    def L(self) -> torch.Tensor:
        """The lower triangular matrix $\mathbf{L}$ of the layers LU decomposition"""
        return self.L_raw.tril(-1)  + torch.eye(self.in_channels).to(self.L_raw.device)

    def U(self) -> torch.Tensor:
        """The upper triangular matrix $\mathbf{U}$ of the layers LU decomposition"""
        return self.U_raw.triu()
    
    def _affine_transform(
        self,
        x: torch.Tensor,
        rev: bool = False
    ) ->  tuple[Any, float | torch.Tensor]:
        """Perform the bijective affine transform.

        Returns transformed outputs and the LogJacDet of the operation.

        Args:
            x (torch.Tensor): Input tensor
            rev (bool, optional): Reverse transformation. Defaults to False.

        Returns:
            tuple[Any, float | torch.Tensor]: Transformed outputs and the LogJacDet of the transformation.
        """
        LU_log_abs_det_jac = torch.diag(self.U()).abs().log().sum()
        if rev:
            L_inv = torch.inverse(self.L())
            U_inv = torch.inverse(self.U())
            weight = torch.matmul(U_inv, L_inv)
        else:
            weight = torch.matmul(self.L(), self.U())
            
        weight = weight.view(
            self.in_channels,
            self.in_channels,
            *([1] * self.input_rank)
        )
        bias = self.LU_bias
        
        if rev:
            bias = bias.view(
                self.in_channels,
                *([1] * self.input_rank)
            )
            
            return (
                self.permute_function(x - bias, weight),
                -LU_log_abs_det_jac
            )

        return (
            self.permute_function(x, weight, bias),
            LU_log_abs_det_jac
        )
        
        
        

    def _pre_permute(self, x: torch.Tensor, rev: bool = False) -> torch.Tensor:
        """Permute before coupling block.

        Only used if ``reverse_permutation`` is True.

        Args:
            x (torch.Tensor): Input tensor
            rev (bool, optional): Reverse the permutation. Defaults to False.

        Returns:
            torch.Tensor: Permuted tensor
        """
        if rev:
            return self.permute_function(x, self.w_perm)

        return self.permute_function(x, self.w_perm_inv)

    def _affine(self, x: torch.Tensor, a: torch.Tensor, rev: bool = False) -> tuple[Any, torch.Tensor]:
        """Perform affine coupling operation.

        Args:
            x (torch.Tensor): Input tensor (passive half)
            a (torch.Tensor): Coupling network outputs
            rev (bool, optional): Reverse the operation. Defaults to False.

        Returns:
            tuple[Any, torch.Tensor]: Transformed tensor and LogJacDet
        """
        # the entire coupling coefficient tensor is scaled down by a
        # factor of ten for stability and easier initialization.
        a *= 0.1
        ch = x.shape[1]

        sub_jac = self.clamp * torch.tanh(a[:, :ch])
        if self.GIN:
            sub_jac -= torch.mean(sub_jac, dim=self.sum_dims, keepdim=True)

        if not rev:
            return (x * torch.exp(sub_jac) + a[:, ch:], torch.sum(sub_jac, dim=self.sum_dims))

        return ((x - a[:, ch:]) * torch.exp(-sub_jac), -torch.sum(sub_jac, dim=self.sum_dims))
    
    def _additive(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        rev: bool = False
    ) -> tuple[Any, torch.Tensor]:
        """Perform additive coupling operation.

        Given the passive half, and the pre-activation outputs of the
        coupling subnetwork, perform the addtive coupling operation.
        Returns both the transformed inputs and the LogJacDet.
        """
        # the entire coupling coefficient tensor is scaled down by a
        # factor of ten for stability and easier initialization.
        a *= 0.1
        ch = x.shape[1]

        if not rev:
            return (x + a[:, ch:], 0.)

        return ((x - a[:, ch:]), 0.)
        
    def _logprior_grad_corrector(self):
        """Computes the gradient corrector term based on a lognormal prior
        on the diagonal of the U matrix of the LU transform"""
        if not self.use_LU:
            corrector = 0
        else:
            corrector = - 2*self.U_raw.diag().abs().log() / self.U_raw.diag() 
            corrector += -1/self.U_raw.diag()
            corrector = corrector.diag()
            
        return corrector  
        
    
    def log_prior(self, correlated: bool = False) -> torch.Tensor:
        """Returns the log prior of the model parameters. If LU layers are used,
        we directly regularize the Jacobean determinant of the flow by putting an
        independent mirrored log-normal
        prior on the diagonal elements of $U$ matrices. The normal has
        mean $0$ and standard deviation $\sqrt{d\cdot #layers}\sigma$, where $d$ is the data dimension.
        That means that we put a log-normal prior on the determinant of the Jacobian.

        Any additive constant is dropped in the optimization procedure.
        """
        if self.use_LU and self.prior_scale is not None:
            log_prior = 0
            for p in self.lu_layers:
                precision = None
                d = self.input_dim
                if correlated:

                    # Pairwise negative correlation of 1/d
                    covariance = -1 / d * torch.ones(d, d).to(self.device) + (1 + 1 / d) * torch.diag(
                        torch.ones(d).to(self.device)
                    )
                    # Scaling
                    covariance = covariance * (self.prior_scale**2)
                else:
                    covariance = torch.eye(d).to(self.device)
                    # Scaling
                    covariance = covariance * (self.prior_scale**2)

                precision = torch.linalg.inv(covariance).to(self.device)

                # log-density of Normal in log-space
                x = p.U.diag().abs().log() 
                log_prior += -(x * (precision @ x)).sum()
                # Change of variables to input space
                log_prior += -x.sum()
            return log_prior
        else:
            return 0
    
    def forward(
        self,
        x: torch.Tensor,
        c: list | None = None,
        rev: bool = False,
        jac: bool = True,
    ) -> tuple[tuple[torch.Tensor], torch.Tensor]:
        """Forward pass through the invertible block.

        Args:
            x (torch.Tensor): Input tensor
            c (list, optional): Conditioning tensors. Defaults to None.
            rev (bool, optional): Reverse the flow. Defaults to False.
            jac (bool, optional): Compute Jacobian determinant. Defaults to True.

        Returns:
            tuple[tuple[torch.Tensor], torch.Tensor]: Tuple of (output tensors,
                LogJacDet)
        """
        del jac  # Unused argument.

        if c is None:
            c = []

        global_scaling_jac = 0
        if self.permute:
            if self.householder:
                self.w_perm = self._construct_householder_permutation()
                if rev or self.reverse_pre_permute:
                    self.w_perm_inv = self.w_perm.transpose(0, 1).contiguous()

            if rev:
                x, scaling_jac = self._permute(x[0], rev=True)
                global_scaling_jac += scaling_jac
                
                x = (x,)
            elif self.reverse_pre_permute:
                x = (self._pre_permute(x[0], rev=False),)
        
        if self.bijective_affine_transform:
            if rev or self.reverse_bijective_affine_transform:
                x, scaling_jac = self._affine_transform(x[0], rev=True)
                global_scaling_jac += scaling_jac
                x = (x,)

        x1, x2 = torch.split(x[0], self.splits, dim=1)

        x1c = torch.cat([x1, *c], 1) if self.conditional else x1

        if self.affine_coupling:
            coupling = self._affine
        else:
            coupling = self._additive
            
        if not rev:
            a1 = self.subnet(x1c)
            x2, j2 = coupling(x2, a1)
        else:
            a1 = self.subnet(x1c)
            x2, j2 = coupling(x2, a1, rev=True)

        log_jac_det = j2
        x_out = torch.cat((x1, x2), 1)
        
        if self.bijective_affine_transform:
            if not rev or self.reverse_bijective_affine_transform:
                x_out, scaling_jac = self._affine_transform(x_out, rev=False)
                global_scaling_jac += scaling_jac

        if self.permute:
            if not rev:
                x_out, global_scaling_jac = self._permute(x_out, rev=False)
            elif self.reverse_pre_permute:
                x_out = self._pre_permute(x_out, rev=True)

        # add the global scaling Jacobian to the total.
        # trick to get the total number of non-channel dimensions:
        # number of elements of the first channel of the first batch member
        n_pixels = x_out[0, :1].numel()
        log_jac_det += (-1) ** rev * n_pixels * global_scaling_jac

        return (x_out,), log_jac_det

    @staticmethod
    def output_dims(input_dims: list[tuple[int]]) -> list[tuple[int]]:
        """Get output dimensions of the layer.

        Args:
            input_dims (list[tuple[int]]): Input dimensions

        Returns:
            list[tuple[int]]: Output dimensions
        """
        return input_dims
    
    

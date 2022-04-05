# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
from typing import Union

# --------------------------------------
import torch as pt

# --------------------------------------
import numpy as np

# --------------------------------------
from pyrception.aux.types import Param
from pyrception.visual.receptor import ReceptorLayer


class BipolarLayer:

    """
    A field of bipolar cells.
    This layer processes the signal form the receptor layer
    and passes it on to the RGC layer.
    """

    def __init__(
        self,
        source: Union[pt.Tensor, ReceptorLayer],
        k_min: int = 1,
        k_max: int = 5,
        sd_w: int = 1 / 4,
        sd_h: int = 1 / 4,
        *args,
        **kwargs,
    ):

        # Initialise the base
        super().__init__(*args, **kwargs)

        # Store the source
        self.source = source

        # Bipolar cell field.
        self.rf = self._make_rf(k_min, k_max, sd_w, sd_h)

    def _make_rf(
        self,
        k_min: int = 1,
        k_max: int = 15,
        sd_w: float = 1 / 3,
        sd_h: float = 1 / 3,
    ):
        def gaussian_2d(
            x: float,
            y: float,
            mx: float,
            my: float,
            sx: float,
            sy: float,
        ):
            # Unnormalised 2D Gaussian.
            return pt.exp(
                -((x - mx) ** 2 / (2 * sx ** 2) + (y - my) ** 2 / (2 * sy ** 2))
            )

        # src = pt.ones(h, w)
        # plt.imshow(src)

        # shape = np.array(src.shape)
        # print(f"==[ shape: {shape}")

        # centre = shape // 2
        # print(f"==[ centre: {centre}")

        # max_side = max(shape)
        # print(f"==[ max_side: {max_side}")

        # Meshgrid of (x, y) coordinates for each pixel
        x = pt.linspace(0, self.dim.padded.W - 1, self.dim.padded.W)
        y = pt.linspace(0, self.dim.padded.H - 1, self.dim.padded.H)
        x, y = pt.meshgrid(x, y, indexing="ij")

        # Unnormalised 2D Gaussian representing the distribution
        # of kernel sizes away from the fovea.
        z = gaussian_2d(
            x,
            y,
            mx=self.dim.padded.W // 2,
            my=self.dim.padded.H // 2,
            sx=self.dim.W * sd_w,
            sy=self.dim.H * sd_h,
        )
        # 　Multiply　by　the maximum kernel size
        z *= k_max

        # Stochastic blurring of the 'edges' created by
        # jumps in the kernel size
        blur = pt.bernoulli(z - pt.floor(z))
        z = pt.ceil(z + blur).int()
        # print(f'==[ blur: {blur}')

        # Set kernel sizes smaller than the minimum to the minimum
        z[pt.where(z < k_min)] = k_min
        # plt.pcolormesh(x, y, z)

        # Compute indices by kernel size
        indices = {}
        for i in range(z.min(), z.max() + 1):
            indices[i] = pt.where(z == i)

        rf_rows = []
        rf_cols = []
        rf_vals = []

        # printksize = _k_min + 1
        printksize = 0

        for ksize, (cols, rows) in indices.items():

            diff = pt.linspace(-ksize // 2 + 1, ksize // 2, ksize)
            # print(f'==[ diff:\n{diff}')

            colspan = (
                ((cols[None, :].repeat(ksize, 1) + diff[:, None]))
                .t()
                .repeat_interleave(ksize, dim=1)
                .int()
            )
            rowspan = (
                (rows[None, :].repeat(ksize, 1) + diff[:, None])
                .t()
                .repeat(1, ksize)
                .int()
            )

            # if ksize == printksize:
            #     print(f"==[ colspan:\n{colspan}")
            #     print(f"==[ rowspan:\n{rowspan}")

            col_idx = colspan * self.dim.padded.H + rowspan

            row_idx = (
                pt.linspace(0, col_idx.shape[0] - 1, col_idx.shape[0])
                .int()[:, None]
                .repeat(1, ksize ** 2)
            )

            # Boundary check
            mask = (
                rowspan.ge(0)
                * rowspan.lt(self.dim.padded.H)
                * colspan.ge(0)
                * colspan.lt(self.dim.padded.W)
            ).flatten()

            if ksize == printksize:
                print(f"==[ mask: {mask}")

            col_idx = col_idx.flatten()
            row_idx = row_idx.flatten()
            # print(f'==[ col_idx:\n{col_idx}')
            # print(f'==[ row_idx:\n{row_idx}')

            sp_rows = (
                (cols * self.dim.padded.H + rows)
                .int()[:, None]
                .repeat(1, ksize ** 2)
                .flatten()
            )
            # print(f"==[ kernel {ksize} ] sp_rows:\n{sp_rows}")

            combined = pt.cat((col_idx[:, None], sp_rows[:, None]), dim=1)

            masked = combined[mask]
            if ksize == printksize:
                print(f"==[ combined:\n{combined}")
                print(f"==[ masked:\n{masked}")
            # print(f"==[ masked shape: {masked.shape}")

            rf_cols.extend(masked[:, 0].tolist())
            rf_rows.extend(masked[:, 1].tolist())
            rf_vals.extend([1 / ksize ** 2] * masked.shape[0])

        # print(f"==[ rf_cols length: {len(rf_cols)}")
        # print(f"==[ rf_rows length: {len(rf_rows)}")
        # print(f"==[ rf_vals length: {len(rf_vals)}")

        rf = (
            pt.sparse_coo_tensor(
                np.array(
                    [
                        rf_rows,
                        rf_cols,
                    ]
                ),
                np.array(rf_vals),
                size=(
                    self.dim.padded.span,
                    self.dim.padded.span,
                ),
                dtype=pt.float32,
            )
            .coalesce()
            .to_sparse_csr()
        )

        # print(f"==[ rf size: {rf.size()}")
        # print(f"==[ rf nnz: {rf._nnz()}")

        return rf

    # ------------------------------------------------------------------------------
    # From Layer
    # ------------------------------------------------------------------------------

    # def __init__(
    #     self,
    #     _input_shape: Tuple[int],
    #     _output_shape: Tuple[int],
    #     _min_weight: float = -1.0,
    #     _max_weight: float = 1.0,
    #     _tau: Union[pt.Tensor, Tuple[float, float]] = (5.0, 50.0),
    #     _threshold_alpha: Optional[float] = None,
    #     _learn: bool = True,
    #     _track: Optional[Set[Param]] = None,
    #     _sparse: bool = False,
    #     _learning_method: LearningMethod = LearningMethod.STDP,
    #     _learning_rate: Optional[float] = None,
    #     _name: str = "",
    # ):

    #     # *  Store init parameters  * #

    #     self.min_weight = _min_weight
    #     self.max_weight = _max_weight

    #     self.input_shape = _input_shape
    #     self.output_shape = _output_shape

    #     # Matrix multiplication operation
    #     # TODO: Implement sparse matrix multiplication (uncomment second line below)
    #     self.matmul_op = pt.matmul
    #     # self.matmul_op = pt.matmul if not _sparse else pt.smm

    #     # Layer name
    #     self.name = _name

    #     # Learning switch.
    #     self.learn = _learn

    #     # Learning toggle.
    #     self.learning_method = _learning_method
    #     self.learning_rate = _learning_rate

    #     self.stdp_steps = 0

    #     # *  Layer parameters  * #
    #     # Membrane time constants.
    #     # This is the only parameter defining the neuron state.
    #     if isinstance(_tau, tuple):
    #         if len(_tau) != 2:
    #             raise ValueError(
    #                 "Please provide a tuple of exactly two floatint-point numbers."
    #             )

    #         tau_min = _tau[0]
    #         tau_max = _tau[1]
    #         _tau = pt.reshape(
    #             pt.linspace(tau_min, tau_max, pt.prod(_output_shape)), _output_shape
    #         )

    #     elif isinstance(_tau, pt.Tensor):

    #         # Sanity check
    #         if pt.count_nonzero(pt.any(_tau <= 0.0)).item() > 0:
    #             raise ValueError(
    #                 "The membrane time constant for each neuron must be greater than 0."
    #             )

    #     self.tau = _tau

    #     # "Forgetting rate" for the membrane potential EMA / EMV of each neuron.
    #     # Also used for computing the decay of the membrane potential
    #     # and the activation in the absence of input.
    #     # Adding 1.0 is a sanity check to ensure that alpha
    #     # is less than 1.
    #     self.alpha = 1.0 / (_tau + 1.0)

    #     # Threshold time constant.
    #     # The threshold adapts much faster than the membrane potential.
    #     # TODO: Separate values for the threshold and activation forgetting rates.

    #     if _threshold_alpha is None:
    #         # The threshold
    #         _threshold_alpha = 10 * self.alpha

    #     self.threshold_alpha = _threshold_alpha
    #     self.activation_alpha = _threshold_alpha

    #     # A layer consists of an array of stateful neurons,
    #     # each with a dynamic membrane potential.
    #     self.potentials = pt.zeros(_output_shape)

    #     # Baseline (a tensor of 0s) used for resetting potentials and activations.
    #     self.baseline = pt.zeros_like(self.potentials)

    #     # Exponential moving mean and variance for the membrane potentials.
    #     self.potential_mean = pt.zeros_like(self.potentials)
    #     self.potential_var = pt.zeros_like(self.potentials)

    #     # Neuron activations.
    #     # These decay gradually at the same rate as the membrane potential.
    #     self.activations = pt.zeros_like(self.potentials)

    #     self.stdp_steps = 0

    #     # Weights for connections between the preceding layer and the current layer.
    #     # TODO: Sparse weights
    #     self.weights = pt.FloatTensor(
    #         [np.prod(_output_shape), np.prod(_input_shape)]
    #     ).uniform_(
    #         _min_weight,
    #         _max_weight,
    #     )

    #     # Parameter dictionary
    #     self.params = {
    #         Param.Input: lambda: None,  # Return the input tensor
    #         Param.Tau: lambda: self.tau,
    #         Param.Alpha: lambda: self.alpha,
    #         Param.ThresholdAlpha: lambda: self.threshold_alpha,
    #         Param.ActivationAlpha: lambda: self.activation_alpha,
    #         Param.Potentials: lambda: self.potentials,
    #         Param.PotentialMean: lambda: self.potential_mean,
    #         Param.PotentialVar: lambda: self.potential_var,
    #         Param.Activations: lambda: self.activations,
    #         Param.Weights: lambda: self.weights,
    #     }

    #     # Parameters to keep track of
    #     self.track = _track if _track is not None else set()

    #     # Parameter history
    #     self.history = {t: [] for t in self.track}

    def _get_param(self, _param: Param):

        return self.params[_param]()

    def _update_potential_stats(self):

        """
        Compute the running mean and variance of the membrane potentials.
        """

        diff = self.potentials - self.potential_mean
        inc = self.alpha * diff
        self.potential_mean += inc
        self.potential_var = (1.0 - self.alpha) * (self.potential_var + diff * inc)

    def _norm_deviation(self):

        """
        Normalised deviation from the mean membrane potential for each neuron.
        """

        # Use nan_to_num_() to avoid crashing if the SD is 0.
        return (self.potentials - self.potential_mean) / pt.sqrt(
            self.potential_var
        ).nan_to_num_()

    def _decay(self):

        """
        Membrane repolarisation (membrane potential decay) and activation decay.
        """

        # Exponential decay of the membrane potential and the total activation.
        self.potentials *= pt.exp(-self.alpha)
        self.activations *= pt.exp(-self.activation_alpha)

        # Update the mean potential.
        self._update_potential_stats()

    def _activate(self):

        """
        The normalised deviation ND is defined by the mean (V_{\mu}) and standard deviation (V_{\sigma})
        values of the membrane potential of the respective neuron:

            ND = \frac{V - V_{\mu}}{V_{\sigma}}

        The activation profile of a neuron is in the shape of a sigmoid (tanh), which approaches 1 asymptotically:

            \rho = \tanh(norm_diff)

        The activation threshold \theta is defined as follows:

            \theta = \exp(- \threshold_alpha * norm_diff)

        Neurons produce action potentials at a (normalised) rate \eta when the potential crosses the activation threshold, i.e., when

            \eta = \rho - \theta > 0
        """

        # Compute the normalised deviation from the mean.
        norm_dev = self._norm_deviation()

        # Update the moving stats for the membrane potentials.
        self._update_potential_stats()

        # Compute the new activations
        rho = pt.tanh(norm_dev)
        activations = self.op(rho, self.baseline)
        self.activations = pt.where(activations, rho, self.activations)

        # Reset membrane potentials to 0 for neurons that have produced action potentials.
        # The mean membrane potential will be updated at the next step.
        self.potentials = pt.where(activations, self.baseline, self.potentials)

    # * Protected methods * #

    def _update_history(self):

        for param in self.track:
            self.history[param].append(self.params[param]())

    # * Public methods * #

    def forward(
        self,
        _input: pt.Tensor,
    ):

        """
        - Integrate incoming signals
        - Update input statistics
        - Compute activations
        """

        # Membrane potential and activity decay.
        if self.tau is not None:
            self._decay()

        # Compute the input potentials from the input signals scaled by the synaptic weights.
        self.potentials += self.mm_op(self.weights, _input)

        # Compute the activation.
        self._activate()

        # Store all tracked parameters.
        self._update_history()

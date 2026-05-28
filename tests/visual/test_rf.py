import pytest
import numpy as np
from scipy.sparse import csr_array, csc_array

from pyrception.visual.rf import ReceptiveFields
from pyrception.utils.enums import RFArrangement, KernelShape, KernelFilter


@pytest.fixture
def logpolar_rf():
    return ReceptiveFields(size=(64, 64), sectors=32)


@pytest.fixture
def cartesian_rf():
    return ReceptiveFields(
        size=(64, 64),
        arrangement=RFArrangement.Cartesian,
        ksize=(5, 5),
    )


class TestReceptiveFieldsLogPolar:
    """Tests for ReceptiveFields with log-polar arrangement."""

    def test_cell_count_is_positive(self, logpolar_rf):
        assert logpolar_rf.cell_count > 0

    def test_cell_coordinates_within_bounds(self, logpolar_rf):
        assert logpolar_rf.cell_coordinates[:, 0].min() >= 0
        assert logpolar_rf.cell_coordinates[:, 0].max() < 64
        assert logpolar_rf.cell_coordinates[:, 1].min() >= 0
        assert logpolar_rf.cell_coordinates[:, 1].max() < 64

    def test_forward_synapses_shape(self, logpolar_rf):
        assert logpolar_rf.forward_synapses.shape[0] == logpolar_rf.cell_count
        assert logpolar_rf.forward_synapses.shape[1] == 64 * 64

    def test_forward_synapses_are_sparse(self, logpolar_rf):
        assert isinstance(logpolar_rf.forward_synapses, csr_array)

    def test_feedback_synapses_shape(self, logpolar_rf):
        assert logpolar_rf.feedback_synapses.shape == (64 * 64, logpolar_rf.cell_count)

    def test_feedback_synapses_are_sparse(self, logpolar_rf):
        assert isinstance(logpolar_rf.feedback_synapses, csc_array)

    def test_kernels_populated(self, logpolar_rf):
        assert len(logpolar_rf.kernels) == logpolar_rf.cell_count

    def test_forward_synapse_rows_sum_to_one(self, logpolar_rf):
        row_sums = logpolar_rf.forward_synapses.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_more_sectors_produces_more_cells(self):
        rf_few = ReceptiveFields(size=(64, 64), sectors=16)
        rf_many = ReceptiveFields(size=(64, 64), sectors=64)
        assert rf_many.cell_count > rf_few.cell_count

    def test_larger_field_produces_more_cells(self):
        rf_small = ReceptiveFields(size=(32, 32), sectors=32)
        rf_large = ReceptiveFields(size=(128, 128), sectors=32)
        assert rf_large.cell_count > rf_small.cell_count

    def test_phyllotactic_produces_valid_output(self):
        rf = ReceptiveFields(size=(64, 64), sectors=32, phyllotactic=True)
        assert rf.cell_count > 0


class TestReceptiveFieldsCartesian:
    """Tests for ReceptiveFields with Cartesian arrangement."""

    def test_cell_count_is_positive(self, cartesian_rf):
        assert cartesian_rf.cell_count > 0

    def test_cell_coordinates_within_bounds(self, cartesian_rf):
        assert cartesian_rf.cell_coordinates[:, 0].min() >= 0
        assert cartesian_rf.cell_coordinates[:, 0].max() < 64
        assert cartesian_rf.cell_coordinates[:, 1].min() >= 0
        assert cartesian_rf.cell_coordinates[:, 1].max() < 64

    def test_forward_synapses_shape(self, cartesian_rf):
        assert cartesian_rf.forward_synapses.shape[0] == cartesian_rf.cell_count
        assert cartesian_rf.forward_synapses.shape[1] == 64 * 64

    def test_smaller_kernel_produces_more_cells(self):
        rf_large_k = ReceptiveFields(
            size=(64, 64),
            arrangement=RFArrangement.Cartesian,
            ksize=(9, 9),
        )
        rf_small_k = ReceptiveFields(
            size=(64, 64),
            arrangement=RFArrangement.Cartesian,
            ksize=(3, 3),
        )
        assert rf_small_k.cell_count > rf_large_k.cell_count


class TestReceptiveFieldsFilters:
    """Tests for different kernel filter types."""

    def test_gaussian_filter_weights_sum_to_one(self):
        rf = ReceptiveFields(size=(32, 32), sectors=16, kfilter=KernelFilter.Gaussian)
        row_sums = rf.forward_synapses.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_uniform_filter_weights_sum_to_one(self):
        rf = ReceptiveFields(size=(32, 32), sectors=16, kfilter=KernelFilter.Uniform)
        row_sums = rf.forward_synapses.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


class TestReceptiveFieldsKernelShapes:
    """Tests for different kernel shapes."""

    def test_rectangular_kernel(self):
        rf = ReceptiveFields(size=(32, 32), sectors=16, kshape=KernelShape.Rectangular)
        assert rf.cell_count > 0
        assert len(rf.kernels) == rf.cell_count

    def test_elliptic_kernel(self):
        rf = ReceptiveFields(size=(32, 32), sectors=16, kshape=KernelShape.Elliptic)
        assert rf.cell_count > 0
        assert len(rf.kernels) == rf.cell_count

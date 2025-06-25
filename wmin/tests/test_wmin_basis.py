"""
Unit tests for wmin.basis.py module

This module contains unit tests for all functions in the wmin.basis module.
Tests are written using only functions without classes.
"""

import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock
import numpy as np

# Mock the imports that might not be available in test environment
import sys
from unittest.mock import MagicMock

# Mock external dependencies
sys.modules["colibri.constants"] = MagicMock()
sys.modules["colibri.export_results"] = MagicMock()
sys.modules["n3fit.model_gen"] = MagicMock()
sys.modules["wmin.utils"] = MagicMock()

# Now import the module under test
from wmin.basis import (
    n3fit_pdf_model,
    get_X_matrix,
    pod_basis,
    write_pod_basis,
)


def create_test_pdf_grid():
    """Helper function to create test PDF grid data."""
    # Shape: (nreplicas=3, nflavours=2, nx=4)
    return np.array(
        [
            [[1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 1.5, 2.0]],  # replica 1
            [[1.1, 2.1, 3.1, 4.1], [0.6, 1.1, 1.6, 2.1]],  # replica 2
            [[0.9, 1.9, 2.9, 3.9], [0.4, 0.9, 1.4, 1.9]],  # replica 3
        ]
    )


def test_n3fit_pdf_model():
    """Test n3fit_pdf_model function."""
    print("Testing n3fit_pdf_model...")

    with patch("wmin.basis.pdfNN_layer_generator") as mock_generator:
        mock_model = MagicMock()
        mock_generator.return_value = mock_model

        # Test with default parameters
        result = n3fit_pdf_model()

        # Verify the function was called with expected parameters
        mock_generator.assert_called_once()
        args, kwargs = mock_generator.call_args

        assert kwargs["nodes"] == [25, 20, 8]
        assert kwargs["activations"] == ["tanh", "tanh", "linear"]
        assert kwargs["impose_sumrule"] == True
        assert kwargs["num_replicas"] == 5  # max_replica - min_replica + 1

        # Test with custom parameters
        custom_params = {
            "replica_range_settings": {"min_replica": 2, "max_replica": 8},
            "nodes": [10, 5],
            "activations": ["relu", "sigmoid"],
        }

        mock_generator.reset_mock()
        result = n3fit_pdf_model(**custom_params)

        args, kwargs = mock_generator.call_args
        assert kwargs["nodes"] == [10, 5]
        assert kwargs["activations"] == ["relu", "sigmoid"]
        assert kwargs["num_replicas"] == 7  # 8 - 2 + 1


def test_get_X_matrix():
    """Test get_X_matrix function."""

    pdf_grid = create_test_pdf_grid()
    X, phi0 = get_X_matrix(pdf_grid)

    # Check shapes
    expected_ndata = pdf_grid.shape[1] * pdf_grid.shape[2]  # nflavours * nx = 2 * 4 = 8
    expected_nfeatures = pdf_grid.shape[0]  # nreplicas = 3

    assert X.shape == (expected_ndata, expected_nfeatures)
    assert phi0.shape == (expected_ndata, 1)

    # Check that mean is correctly subtracted
    # The mean of X along axis=1 should be close to zero
    row_means = np.mean(X, axis=1)
    np.testing.assert_allclose(row_means, 0.0, atol=1e-10)

    # Check that phi0 contains the original mean
    original_reshaped = pdf_grid.reshape(pdf_grid.shape[0], -1).T
    expected_phi0 = np.mean(original_reshaped, axis=1)[:, np.newaxis]
    np.testing.assert_allclose(phi0, expected_phi0)


def test_pod_basis():
    """Test pod_basis function."""
    print("Testing pod_basis...")

    pdf_grid = create_test_pdf_grid()
    Neig = 2

    pod, phi0 = pod_basis(pdf_grid, Neig)

    # Check shapes
    assert pod.shape == (Neig, pdf_grid.shape[1], pdf_grid.shape[2])
    assert phi0.shape == (pdf_grid.shape[1], pdf_grid.shape[2])

    # Check that POD basis has expected properties
    assert isinstance(pod, np.ndarray)
    assert isinstance(phi0, np.ndarray)

    # Test with different Neig values
    for neig in [1, 3]:
        if neig <= min(pdf_grid.shape[0], pdf_grid.shape[1] * pdf_grid.shape[2]):
            pod_test, phi0_test = pod_basis(pdf_grid, neig)
            assert pod_test.shape == (neig, pdf_grid.shape[1], pdf_grid.shape[2])


def test_write_pod_basis():
    """Test write_pod_basis function."""
    print("Testing write_pod_basis...")

    with patch("wmin.basis.write_exportgrid") as mock_write:
        with patch("wmin.basis.os.path.exists", return_value=False):
            with patch("wmin.basis.os.mkdir") as mock_mkdir:

                # Create test data
                pod = np.random.rand(3, 2, 4)  # (Neig, nflavours, nx)
                phi0 = np.random.rand(2, 4)  # (nflavours, nx)
                pod_basis_data = (pod, phi0)

                temp_dir = tempfile.mkdtemp()
                output_path = os.path.join(temp_dir, "test_fit")

                try:
                    write_pod_basis(pod_basis_data, output_path)

                    # Check that directories were created
                    mock_mkdir.assert_called()

                    # Check that write_exportgrid was called correct number of times
                    assert mock_write.call_count == pod.shape[0]

                    # Check first call (central member)
                    first_call = mock_write.call_args_list[0]
                    args, kwargs = first_call
                    np.testing.assert_array_equal(kwargs["grid_for_writing"], phi0)
                    assert kwargs["replica_index"] == 1

                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)

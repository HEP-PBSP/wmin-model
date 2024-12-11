"""
wmin.tests.test_wmin_basis.py

Test the wmin.basis module.
"""

import pytest
import shutil
import pathlib
import numpy as np
import unittest
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_allclose
import os

from wmin.basis import (
    wmin_pdfbasis_normalization,
    wmin_basis_sum_rules_normalization,
    sum_rules_dict,
    wmin_basis_replica_selector,
    wmin_basis_pdf_grid,
    _get_X_exportgrids,
    mc2_pca,
    write_pca_basis_exportgrids,
)
from colibri.constants import FLAVOUR_TO_ID_MAPPING, LHAPDF_XGRID, EXPORT_LABELS
from colibri.tests.conftest import TEST_PDFSET


from validphys.core import PDF
from validphys import convolution


# mock known sum rules
MOCK_KNOWN_SUM_RULES_EXPECTED = {
    "momentum": 1,
    "uvalence": 0.5,
    "dvalence": 0.25,
    "svalence": 0.1,
    "cvalence": 0.05,
}


@patch("wmin.basis.KNOWN_SUM_RULES_EXPECTED", MOCK_KNOWN_SUM_RULES_EXPECTED)
def test_correct_replicas_selected():

    # # Define a dictionary with replica values for each sum rule type
    # mock_known_sum_rules.result = KNOWN_SUM_RULES_EXPECTED

    sum_rule_dict = {
        "momentum": np.array([1.0, 0.98, 1.01, 1.02]),
        "uvalence": np.array([0.5, 0.52, 0.49, 0.5]),
        "dvalence": np.array([0.25, 0.26, 0.24, 0.25]),
        "svalence": np.array([0.1, 0.1, 0.11, 0.09]),
        "cvalence": np.array([0.05, 0.04, 0.05, 0.05]),
    }

    expected_indices = np.array(
        [0, 2]
    )  # These replicas should match within default tolerance
    selected_indices = wmin_basis_replica_selector(sum_rule_dict)
    np.testing.assert_array_equal(selected_indices, expected_indices)


@patch("wmin.basis.KNOWN_SUM_RULES_EXPECTED", MOCK_KNOWN_SUM_RULES_EXPECTED)
def test_no_replicas_selected():
    sum_rule_dict = {
        "momentum": np.array([0.9, 0.95, 0.8]),
        "uvalence": np.array([0.4, 0.3, 0.35]),
        "dvalence": np.array([0.2, 0.1, 0.15]),
        "svalence": np.array([0.05, 0.07, 0.03]),
        "cvalence": np.array([0.02, 0.01, 0.03]),
    }

    selected_indices = wmin_basis_replica_selector(sum_rule_dict)
    assert len(selected_indices) == 0  # Expect no replicas to pass


@patch("wmin.basis.KNOWN_SUM_RULES_EXPECTED", MOCK_KNOWN_SUM_RULES_EXPECTED)
def test_all_replicas_selected():
    sum_rule_dict = {
        "momentum": np.array([1.0, 1.0, 1.0]),
        "uvalence": np.array([0.5, 0.5, 0.5]),
        "dvalence": np.array([0.25, 0.25, 0.25]),
        "svalence": np.array([0.1, 0.1, 0.1]),
        "cvalence": np.array([0.05, 0.05, 0.05]),
    }

    expected_indices = np.array([0, 1, 2])
    selected_indices = wmin_basis_replica_selector(sum_rule_dict)
    np.testing.assert_array_equal(selected_indices, expected_indices)


@patch("wmin.basis.KNOWN_SUM_RULES_EXPECTED", MOCK_KNOWN_SUM_RULES_EXPECTED)
def test_different_tolerance():
    sum_rule_dict = {
        "momentum": np.array([1.0, 0.98, 1.01, 1.02]),
        "uvalence": np.array([0.5, 0.52, 0.49, 0.5]),
        "dvalence": np.array([0.25, 0.26, 0.24, 0.25]),
        "svalence": np.array([0.1, 0.1, 0.11, 0.09]),
        "cvalence": np.array([0.05, 0.04, 0.05, 0.05]),
    }

    selected_indices_default_tol = wmin_basis_replica_selector(sum_rule_dict)
    assert len(selected_indices_default_tol) == 2  # Should return some replicas

    selected_indices_high_tol = wmin_basis_replica_selector(
        sum_rule_dict, sum_rule_atol=0.05
    )
    np.testing.assert_array_equal(
        selected_indices_high_tol, np.array([0, 1, 2, 3])
    )  # All replicas pass with high tolerance


@pytest.mark.parametrize("pdf_basis", ["intrinsic_charm", "perturbative_charm"])
def test_wmin_pdfbasis_normalization(pdf_basis):
    """
    Test that the PDF-basis, intrinsic and perturbative charm, normalisation
    works as expected.
    """
    pdf_grid = np.random.rand(100, 14, 50)

    pdf_grid_normalized = wmin_pdfbasis_normalization(
        pdf_grid=pdf_grid, pdf_basis=pdf_basis
    )

    sigma = pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["\Sigma"], :]
    valence = pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["V"], :]

    if pdf_basis == "intrinsic_charm":
        assert_allclose(
            pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["V15"], :], valence
        )
        assert_allclose(
            pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["V24"], :], valence
        )
        assert_allclose(
            pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["V35"], :], valence
        )
        assert_allclose(pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["T24"], :], sigma)
        assert_allclose(pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["T35"], :], sigma)

    elif pdf_basis == "perturbative_charm":
        assert_allclose(
            pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["V15"], :], valence
        )
        assert_allclose(
            pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["V24"], :], valence
        )
        assert_allclose(
            pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["V35"], :], valence
        )
        assert_allclose(pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["T15"], :], sigma)
        assert_allclose(pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["T24"], :], sigma)
        assert_allclose(pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["T35"], :], sigma)


@pytest.mark.parametrize(
    "pdf_basis",
    [
        "random_basis",
    ],
)
def test_wmin_pdfbasis_normalization_raise_error(pdf_basis):
    """
    Test that the PDF-basis, intrinsic and perturbative charm, normalisation
    works as expected.
    """
    pdf_grid = np.random.rand(100, 14, 50)

    with unittest.TestCase().assertRaises(ValueError):
        pdf_grid_normalized = wmin_pdfbasis_normalization(
            pdf_grid=pdf_grid, pdf_basis=pdf_basis
        )


def test_wmin_basis_sum_rules_normalization():
    """
    Test that the PDF basis sum-rules normalisation works as expected.
    """
    pdf = PDF(TEST_PDFSET)
    sr = sum_rules_dict(pdf)[TEST_PDFSET]

    pdf_grid = convolution.evolution.grid_values(
        pdf, convolution.FK_FLAVOURS, LHAPDF_XGRID, [1.65]
    ).squeeze(-1)

    pdf_grid_sr_norm = wmin_basis_sum_rules_normalization(pdf_grid, sum_rule_dict=sr)

    assert_allclose(
        np.trapz(
            pdf_grid_sr_norm[:, FLAVOUR_TO_ID_MAPPING["V"], :] / np.array(LHAPDF_XGRID),
            x=LHAPDF_XGRID,
            axis=-1,
        ),
        3,
        rtol=1e-2,
    )
    assert_allclose(
        np.trapz(
            pdf_grid_sr_norm[:, FLAVOUR_TO_ID_MAPPING["V3"], :]
            / np.array(LHAPDF_XGRID),
            x=LHAPDF_XGRID,
            axis=-1,
        ),
        1,
        rtol=1e-2,
    )
    assert_allclose(
        np.trapz(
            pdf_grid_sr_norm[:, FLAVOUR_TO_ID_MAPPING["V8"], :]
            / np.array(LHAPDF_XGRID),
            x=LHAPDF_XGRID,
            axis=-1,
        ),
        3,
        rtol=1e-2,
    )
    assert_allclose(
        np.trapz(
            (
                pdf_grid_sr_norm[:, FLAVOUR_TO_ID_MAPPING["\Sigma"], :]
                + pdf_grid_sr_norm[:, FLAVOUR_TO_ID_MAPPING["g"], :]
            ),
            x=LHAPDF_XGRID,
            axis=-1,
        ),
        1,
        rtol=1e-2,
    )


import numpy as np
from unittest.mock import patch, MagicMock

# Mock data and values for testing
mock_xgrid = np.array([0.001, 0.01, 0.1, 0.5])  # Example x-grid
mock_pdf_sum_rules = [
    {
        "pdf1": {
            "momentum": np.array([1.0]),
            "uvalence": np.array([0.5]),
            "dvalence": np.array([0.25]),
            "svalence": np.array([0.25]),
        }
    },
    {
        "pdf2": {
            "momentum": np.array([1.0]),
            "uvalence": np.array([0.5]),
            "dvalence": np.array([0.25]),
            "dvalence": np.array([0.25]),
        }
    },
]
mock_selected_replicas = np.array([0, 2])  # Indices of replicas that pass sum rules
mock_pdf_grid = np.random.rand(
    2 * len(mock_selected_replicas), 5, len(mock_xgrid), 1
)  # Example PDF grid


@patch("wmin.basis.convolution.evolution.grid_values")
@patch("wmin.basis.wmin_basis_sum_rules_normalization")
@patch("wmin.basis.wmin_pdfbasis_normalization")
@patch("wmin.basis.wmin_basis_replica_selector")
def test_wmin_basis_pdf_grid_valid_output(
    mock_replica_selector,
    mock_basis_normalization,
    mock_sum_rules_normalization,
    mock_grid_values,
):

    # Mock return values
    mock_replica_selector.return_value = mock_selected_replicas
    mock_grid_values.return_value = np.random.rand(10, 5, len(mock_xgrid), 1)
    mock_sum_rules_normalization.side_effect = (
        lambda grid, **kwargs: grid
    )  # Pass-through
    mock_basis_normalization.side_effect = lambda grid, **kwargs: grid  # Pass-through

    # Run the function
    result = wmin_basis_pdf_grid(
        pdfs_sum_rules=mock_pdf_sum_rules,
        pdf_basis="intrinsic_charm",
        Q=1.65,
        xgrid=mock_xgrid,
    )

    # Check that result shape is as expected
    assert result.shape == (
        len(mock_selected_replicas) * len(mock_pdf_sum_rules),
        5,
        len(mock_xgrid),
    ), "Resulting PDF grid shape is incorrect."


@patch("wmin.basis.wmin_basis_replica_selector")
def test_wmin_basis_pdf_grid_no_valid_replicas(
    mock_replica_selector,
):
    # Mock replica selector to return empty array, simulating no valid replicas
    mock_replica_selector.return_value = np.array([])

    # Run the function and expect raise error
    with unittest.TestCase().assertRaises(ValueError):
        wmin_basis_pdf_grid(
            pdfs_sum_rules=mock_pdf_sum_rules,
            pdf_basis="intrinsic_charm",
            Q=1.65,
            xgrid=mock_xgrid,
        )


@patch("wmin.basis.convolution.evolution.grid_values")
@patch("wmin.basis.wmin_basis_sum_rules_normalization")
@patch("wmin.basis.wmin_pdfbasis_normalization")
@patch("wmin.basis.wmin_basis_replica_selector")
def test_wmin_basis_pdf_grid_normalization_calls(
    mock_replica_selector,
    mock_sum_rules_normalization,
    mock_basis_normalization,
    mock_grid_values,
):

    # Mock return values
    mock_replica_selector.return_value = mock_selected_replicas
    mock_grid_values.return_value = mock_pdf_grid

    # Run the function
    result = wmin_basis_pdf_grid(
        pdfs_sum_rules=mock_pdf_sum_rules,
        pdf_basis="intrinsic_charm",
        Q=1.65,
        xgrid=mock_xgrid,
    )

    # Check that the sum rule normalization was called with the expected parameters
    mock_sum_rules_normalization.assert_called()
    mock_basis_normalization.assert_called()


def test_get_X_exportgrids():
    """
    Test the _get_X_exportgrids function to ensure it correctly reshapes and subtracts the mean of the pdfgrid.
    """
    # Mock input data
    pdfgrid = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )  # Shape (2, 2, 2)

    # Expected output
    reshaped_pdfgrid = pdfgrid.reshape(2, 4)  # Shape (2, 4)
    mean_subtracted_pdfgrid = reshaped_pdfgrid - reshaped_pdfgrid.mean(axis=0)
    expected_result = mean_subtracted_pdfgrid.T  # Shape (4, 2)
    # Call the function under test
    result = _get_X_exportgrids(pdfgrid)

    # Assertions
    np.testing.assert_array_equal(
        result,
        expected_result,
        "The reshaped and mean-subtracted pdfgrid result is incorrect.",
    )


def test_mc2_pca():
    # Mock dependencies
    mock_pdf = MagicMock()
    mock_Q = MagicMock()
    mock_output_path = pathlib.Path("/mock/output/path")
    mock_gridname = "mock_grid"

    # Mock external functions
    with patch("pathlib.Path") as MockPath, patch(
        "wmin.basis.lhaindex.get_lha_datapath", return_value="/mock/lha/path"
    ), patch("wmin.basis.lhaindex.isinstalled", return_value=False), patch(
        "shutil.copytree"
    ), patch(
        "shutil.rmtree"
    ), patch(
        "wmin.basis._create_mc2pca", return_value="/mock/result/path"
    ) as mock_create_mc2pca, patch(
        "wmin.basis.mc2hessian_xgrid", return_value="mock_xgrid"
    ):

        MockPath.return_value = pathlib.Path("/mock/lha/path")

        # Call the function under test
        mc2_pca(
            pdf=mock_pdf,
            Q=mock_Q,
            Neig=10,
            output_path=mock_output_path,
            gridname=mock_gridname,
            installgrid=True,
            hessian_normalization=True,
        )

        # Assertions for _create_mc2pca
        mock_create_mc2pca.assert_called_once_with(
            mock_pdf,
            Q=mock_Q,
            xgrid="mock_xgrid",
            Neig=10,
            output_path=mock_output_path,
            name=mock_gridname,
            hessian_normalization=True,
        )

        # Assertions for installation logic
        shutil.copytree.assert_called_once_with(
            "/mock/result/path", pathlib.Path("/mock/lha/path") / mock_gridname
        )
        shutil.rmtree.assert_not_called()  # Since isinstalled returns False in this test


def test_write_pca_basis_exportgrids():
    # Mock dependencies
    mock_fit_path = pathlib.Path("/mock/fit/path")
    mock_output_path = pathlib.Path("/mock/output/path")
    mock_pdf_grid = np.random.rand(100, 14, 196)  # Mock PDF grid
    mock_X = np.random.rand(14 * 196, 100)  # Mock X matrix
    mock_V = np.random.rand(100, 3)  # Mock PCA basis vectors

    # Mock external functions
    with patch(
        "wmin.basis.get_pdfgrid_from_exportgrids", return_value=mock_pdf_grid
    ) as mock_get_pdfgrid, patch(
        "wmin.basis._get_X_exportgrids", return_value=mock_X
    ) as mock_get_X, patch(
        "wmin.basis._compress_X", return_value=mock_V
    ) as mock_compress_X, patch(
        "shutil.copy"
    ), patch(
        "os.makedirs"
    ), patch(
        "os.mkdir"
    ), patch(
        "os.path.exists", return_value=False
    ), patch(
        "wmin.basis.write_exportgrid"
    ) as mock_write_exportgrid:

        # Call the function under test
        write_pca_basis_exportgrids(
            fit_path=mock_fit_path,
            Neig=3,
            output_path=mock_output_path,
            hessian_normalization=True,
        )

        # Assertions for get_pdfgrid_from_exportgrids
        mock_get_pdfgrid.assert_called_once_with(mock_fit_path)

        # Assertions for _get_X_exportgrids

        np.testing.assert_array_equal(mock_get_X.call_args[0][0], mock_pdf_grid)
        # mock_get_X.assert_called_once_with(mock_pdf_grid.copy())

        # Assertions for _compress_X
        mock_compress_X.assert_called_once_with(mock_X, 3)

        # Assertions for directory creation
        os.makedirs.assert_any_call(mock_output_path / "input", exist_ok=True)

        # Assertions for shutil.copy
        shutil.copy.assert_called_once_with(
            mock_fit_path / "input/runcard.yaml",
            mock_output_path / "input/runcard.yaml",
        )

        # Assertions for write_exportgrid
        assert mock_write_exportgrid.call_count == 3
        for i in range(3):
            assert (
                mock_write_exportgrid.call_args_list[i][1]["grid_name"]
                == mock_output_path / f"replicas/replica_{i+1}" / mock_output_path.name
            )
            assert mock_write_exportgrid.call_args_list[i][1]["replica_index"] == i + 1
            assert mock_write_exportgrid.call_args_list[i][1]["Q"] == 1.65
            assert mock_write_exportgrid.call_args_list[i][1]["xgrid"] == LHAPDF_XGRID
            assert (
                mock_write_exportgrid.call_args_list[i][1]["export_labels"]
                == EXPORT_LABELS
            )

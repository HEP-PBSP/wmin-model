"""
wmin.tests.test_wmin_basis.py

Test the wmin.basis module.
"""

import pytest
import unittest
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_allclose

from wmin.basis import (
    wmin_pdfbasis_normalization,
    wmin_basis_sum_rules_normalization,
    sum_rules_dict,
    wmin_basis_replica_selector,
    wmin_basis_pdf_grid,
)
from colibri.constants import FLAVOUR_TO_ID_MAPPING, LHAPDF_XGRID
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

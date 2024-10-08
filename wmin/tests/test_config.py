"""
Module for tests of the wmin.config module.
"""

from unittest.mock import patch
from wmin.config import WminConfig
from reportengine.configparser import ConfigError
import unittest

INPUT_PARAMS = {}
CONFIG = WminConfig(INPUT_PARAMS)


@patch("wmin.config.log.warning")
def test_parse_wmin_settings_valid(mock_log_warning):

    settings = {
        "n_basis": 10,
        "wminpdfset": "PDF",
        "wmin_inherited_evolution": True,
    }

    expected = {
        "n_basis": 10,
        "wminpdfset": "PDF",
        "wmin_inherited_evolution": True,
    }

    result = CONFIG.parse_wmin_settings(settings)
    assert result == expected
    mock_log_warning.assert_not_called()


@patch("wmin.config.log.warning")
def test_parse_wmin_settings_missing_n_basis(mock_log_warning):

    settings = {
        "wminpdfset": "PDF",
        "wmin_inherited_evolution": True,
    }

    with unittest.TestCase().assertRaises(ValueError) as context:
        CONFIG.parse_wmin_settings(settings)
    assert str(context.exception) == "Missing key n_basis for wmin_settings"
    mock_log_warning.assert_not_called()


@patch("wmin.config.log.warning")
def test_parse_wmin_settings_missing_wminpdfset(mock_log_warning):

    settings = {
        "n_basis": 10,
        "wmin_inherited_evolution": True,
    }

    with unittest.TestCase().assertRaises(ValueError) as context:
        CONFIG.parse_wmin_settings(settings)
    assert str(context.exception) == "Missing key wminpdfset for wmin_settings"
    mock_log_warning.assert_not_called()


@patch("wmin.config.log.warning")
def test_parse_wmin_settings_with_unknown_key(mock_log_warning):

    settings = {
        "n_basis": 10,
        "wminpdfset": "PDF",
        "wmin_inherited_evolution": True,
        "unknown_key": "value",
    }

    expected = {
        "n_basis": 10,
        "wminpdfset": "PDF",
        "wmin_inherited_evolution": True,
    }

    result = CONFIG.parse_wmin_settings(settings)
    assert result == expected
    mock_log_warning.assert_called_once()
    args, kwargs = mock_log_warning.call_args
    assert isinstance(args[0], ConfigError)
    assert "Key 'unknown_key' in ns_settings not known." in str(args[0])


def test_parse_prior_settings_valid_uniform_with_min_max():
    settings = {"type": "uniform_parameter_prior", "min_val": -5, "max_val": 10}

    expected = {"type": "uniform_parameter_prior", "min_val": -5, "max_val": 10}

    result = CONFIG.parse_prior_settings(settings)
    assert result == expected


def test_parse_prior_settings_valid_uniform_default_min_max():
    settings = {"type": "uniform_parameter_prior"}

    expected = {"type": "uniform_parameter_prior", "min_val": -1, "max_val": 1}

    result = CONFIG.parse_prior_settings(settings)
    assert result == expected


def test_parse_prior_settings_missing_type():
    settings = {"min_val": -5, "max_val": 10}

    with unittest.TestCase().assertRaises(ValueError) as context:
        CONFIG.parse_prior_settings(settings)
    assert str(context.exception) == "Missing key type for prior_settings"


def test_parse_prior_settings_non_uniform_type():
    settings = {"type": "non_uniform_type"}

    expected = {"type": "non_uniform_type"}

    result = CONFIG.parse_prior_settings(settings)
    assert result == expected  # Non-uniform types should remain unmodified

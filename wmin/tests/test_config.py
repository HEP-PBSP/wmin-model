"""
Module for tests of the wmin.config module.
"""

from unittest.mock import patch
from wmin.config import WminConfig
from reportengine.configparser import ConfigError
import unittest


@patch("wmin.config.log.warning")
def test_parse_wmin_settings_valid(mock_log_warning):
    input_params = {}
    config = WminConfig(input_params)

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

    result = config.parse_wmin_settings(settings)
    assert result == expected
    mock_log_warning.assert_not_called()


@patch("wmin.config.log.warning")
def test_parse_wmin_settings_missing_n_basis(mock_log_warning):
    input_params = {}
    config = WminConfig(input_params)

    settings = {
        "wminpdfset": "PDF",
        "wmin_inherited_evolution": True,
    }

    with unittest.TestCase().assertRaises(ValueError) as context:
        config.parse_wmin_settings(settings)
    assert str(context.exception) == "Missing key n_basis for wmin_settings"
    mock_log_warning.assert_not_called()


@patch("wmin.config.log.warning")
def test_parse_wmin_settings_missing_wminpdfset(mock_log_warning):
    input_params = {}
    config = WminConfig(input_params)

    settings = {
        "n_basis": 10,
        "wmin_inherited_evolution": True,
    }

    with unittest.TestCase().assertRaises(ValueError) as context:
        config.parse_wmin_settings(settings)
    assert str(context.exception) == "Missing key wminpdfset for wmin_settings"
    mock_log_warning.assert_not_called()


@patch("wmin.config.log.warning")
def test_parse_wmin_settings_with_unknown_key(mock_log_warning):
    input_params = {}
    config = WminConfig(input_params)

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

    result = config.parse_wmin_settings(settings)
    assert result == expected
    mock_log_warning.assert_called_once()
    args, kwargs = mock_log_warning.call_args
    assert isinstance(args[0], ConfigError)
    assert "Key 'unknown_key' in ns_settings not known." in str(args[0])

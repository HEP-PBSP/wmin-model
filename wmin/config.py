"""
wmin.config.py

Config module of wmin

"""

import dill
from validphys.core import PDF
from wmin.model import WMinPDF
import logging
from reportengine.configparser import ConfigError

from colibri.config import Environment, colibriConfig

log = logging.getLogger(__name__)


class Environment(Environment):
    pass


class WminConfig(colibriConfig):
    """
    WminConfig class Inherits from colibri.config.colibriConfig
    """

    def parse_prior_settings(self, settings):
        """
        Parse the prior settings for the wmin fit.
        """
        if "type" not in settings.keys():
            raise ValueError("Missing key type for prior_settings")

        # Currently, only prior is uniform with max/min val
        if settings["type"] == "uniform_parameter_prior":
            # Check if max and min vals are defined, if not set them to defaults
            # of -1 and 1.
            if "min_val" not in settings.keys():
                settings["min_val"] = -1
            if "max_val" not in settings.keys():
                settings["max_val"] = 1

        return settings

    def parse_wmin_settings(self, settings):
        """
        Parse the wmin settings onto a dictionary.
        """
        known_keys = {"n_basis", "wminpdfset", "wmin_inherited_evolution"}

        kdiff = settings.keys() - known_keys
        for k in kdiff:
            log.warning(
                ConfigError(f"Key '{k}' in ns_settings not known.", k, known_keys)
            )

        wmin_settings = {}

        # Set the ultranest seed
        if "n_basis" not in settings.keys():
            raise ValueError("Missing key n_basis for wmin_settings")
        wmin_settings["n_basis"] = settings.get("n_basis")

        if "wminpdfset" not in settings.keys():
            raise ValueError("Missing key wminpdfset for wmin_settings")
        wmin_settings["wminpdfset"] = settings.get("wminpdfset")

        wmin_settings["wmin_inherited_evolution"] = settings.get(
            "wmin_inherited_evolution", False
        )

        return settings

    def produce_pdf_model(self, wmin_settings, output_path, dump_model=True):
        """
        Weight minimization grid is in the evolution basis.
        The following parametrization is used:

        f_{j,wm} = f_j + sum_i(w_i * (f_i - f_j))

        this has the advantage of automatically satisfying the sum rules.

        Notes:
            - the central replica of the wminpdfset is always included in the
            wmin parametrization
        """

        model = WMinPDF(PDF(wmin_settings["wminpdfset"]), wmin_settings["n_basis"])

        # dump model to output_path using dill
        # this is mainly needed by scripts/ns_resampler.py
        if dump_model:
            with open(output_path / "pdf_model.pkl", "wb") as file:
                dill.dump(model, file)
        return model

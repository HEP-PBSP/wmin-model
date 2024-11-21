"""
wmin.config.py

Config module of wmin

"""

import dill
from validphys.core import PDF
from wmin.model import WMinPDF

from colibri.config import Environment, colibriConfig


class Environment(Environment):
    pass


class WminConfig(colibriConfig):
    """
    WminConfig class Inherits from colibri.config.colibriConfig
    """

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

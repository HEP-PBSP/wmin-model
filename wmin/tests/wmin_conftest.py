"""
Module containing standard pytest data configurations for testing purposes.
"""

from colibri.core import PriorSettings

EXE = "wmin"
POSTFIT_EXE = "mc_postfit"
TEST_PDFSET = "NNPDF40_nnlo_as_01180"

TEST_PRIOR_SETTINGS_WMIN = PriorSettings(
    **{
        "prior_distribution": "uniform_parameter_prior",
        "prior_distribution_specs": {"max_val": 10.0, "min_val": -10.0},
    }
)
TEST_WMIN_SETTINGS_NBASIS_10 = {
    "wmin_settings": {"wminpdfset": TEST_PDFSET, "n_basis": 10}
}

TEST_WMIN_SETTINGS_NBASIS_100 = {
    "wmin_settings": {"wminpdfset": TEST_PDFSET, "n_basis": 100}
}


RUNCARD_WMIN_LIKELIHOOD_TYPE = "wmin_likelihood_type_test.yaml"

RUNCARD_TEST_FIT_WMIN_BAYES_DIS = "test_fit_wmin_bayes_dis_L0.yaml"

RUNCARD_TEST_FIT_WMIN_BAYES_HAD = "test_fit_wmin_bayes_had_L0.yaml"

RUNCARD_TEST_FIT_WMIN_MC_DIS = "test_fit_wmin_mc_dis_L0.yaml"

RUNCARD_TEST_FIT_WMIN_MC_HAD = "test_fit_wmin_mc_had_L0.yaml"

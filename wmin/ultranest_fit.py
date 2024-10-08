"""
wmin.ultranest_fit.py

This module allows to override some of the functions defined in colibri.ultranest_fit.py if necessary.
"""

from wmin.export_results import write_lhapdf_from_ultranest_result


def run_ultranest_fit(
    ultranest_fit,
    output_path,
    pdf_model,
    wmin_settings,
    ns_settings,
    errortype="replicas",
):
    """
    Overrides the run_ultranest_fit function in colibri.ultranest_fit.py.
    It allows to export the results of an Ultranest fit in such a way that no evolution is
    needed on the fitted wmin set once the fit has run. Evolution is directly inherited from
    the basis that was used in the fit.

    Parameters
    ----------
    ultranest_fit: UltranestFit
        The results of the Ultranest fit.
    output_path: pathlib.PosixPath
        Path to the output folder.
    pdf_model: pdf_model.PDFModel
        The PDF model used in the fit.
    wmin_settings: dict
        Dictionary containing the wmin settings.
    ns_settings: dict
        Dictionary containing the settings for the nested sampling fit.
    errortype: str
        The type of error to be calculated. Default is "replicas".
    """

    if wmin_settings["wmin_inherited_evolution"]:
        write_lhapdf_from_ultranest_result(
            wmin_settings,
            ultranest_fit,
            ns_settings,
            output_path,
            errortype,
        )
    else:
        # import here to avoid problems with duplicated names
        from colibri.ultranest_fit import run_ultranest_fit

        return run_ultranest_fit(ultranest_fit, output_path, pdf_model)

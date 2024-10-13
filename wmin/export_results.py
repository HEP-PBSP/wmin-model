"""
wmin.export_results.py

This module contains functions tailored for the export of fit results that are
performed in the wmin parameterization.
"""

import pathlib
import lhapdf
import os
import logging
import numpy as np

from validphys.lhio import (
    load_all_replicas,
    rep_matrix,
    write_replica,
    generate_replica0,
)
from validphys.core import PDF

log = logging.getLogger(__name__)


def write_wmin_combined_replicas(wmin_parameters, replicas_df, new_wmin_pdf):
    """
    Writes a new LHAPDF set from the results of an UltraNest fit.
    The UltraNest fit must have been performed using a wmin parameterization so that the
    new set can be written as a sum rule conserving linear combination of the replicas of
    the basis set.

    Parameters
    ----------
    wmin_parameters: Array
        wmin parameters posterior samples, shape (n_posterior_samples, n_params)

    replicas_df: DataFrame
        DataFrame containing replicas of the basis set at all scales

    new_wmin_pdf: Path
        Path to the new wmin PDF set
    """
    n_params = wmin_parameters.shape[1]

    for i, wmin_weight in enumerate(wmin_parameters):

        wmin_centr_rep, replica = (
            replicas_df.loc[:, [1]],
            replicas_df.loc[:, range(2, n_params + 2)],
        )

        wm_replica = wmin_centr_rep.dot([1.0 - np.sum(wmin_weight)]) + replica.dot(
            wmin_weight
        )

        wm_headers = f"PdfType: replica\nFormat: lhagrid1\nFromMCReplica: {i}\n"
        log.info(f"Writing replica {i + 1} to {new_wmin_pdf}")
        write_replica(i + 1, new_wmin_pdf, wm_headers.encode("UTF-8"), wm_replica)


def write_new_lhapdf_info_file_from_previous_pdf(
    path_old_pdfset,
    name_old_pdfset,
    path_new_pdfset,
    name_new_pdfset,
    num_members,
    description_set="Weight-minimized set",
    errortype="replicas",
):
    """
    Writes a new LHAPDF set info file based on an existing set.
    """

    # write LHAPDF info file for a new pdf set
    with open(path_old_pdfset / f"{name_old_pdfset}.info", "r") as in_stream, open(
        path_new_pdfset / f"{name_new_pdfset}.info", "w"
    ) as out_stream:
        for l in in_stream.readlines():
            if l.find("SetDesc:") >= 0:
                out_stream.write(f'SetDesc: f"{description_set}"\n')
            elif l.find("NumMembers:") >= 0:
                out_stream.write(f"NumMembers: {num_members}\n")
            elif l.find("ErrorType:") >= 0:
                out_stream.write(f"ErrorType: {errortype}\n")
            else:
                out_stream.write(l)
    log.info(f"Info file written to {path_new_pdfset / f'{name_new_pdfset}.info'}")


def write_lhapdf_from_ultranest_result(
    wmin_settings,
    ultranest_fit,
    ns_settings,
    output_path,
    errortype: str = "replicas",
):
    """
    Writes a new LHAPDF set from the results of an UltraNest fit.
    The UltraNest fit must have been performed using a wmin parameterization so that the
    new set can be written as a linear combination of the replicas of the basis set.

    Parameters
    ----------

    """
    wminpdfset = PDF(wmin_settings["wminpdfset"])

    lhapdf_path = pathlib.Path(lhapdf.paths()[-1])

    # path to pdf set that was used as a basis for the wmin fit
    wmin_basis_pdf = lhapdf_path / str(wminpdfset)

    wmin_fit_name = pathlib.Path(output_path).name

    # path to new wmin pdf set
    new_wmin_pdf = lhapdf_path / wmin_fit_name

    # create new wmin pdf set folder in lhapdf path if it does not exist
    if not new_wmin_pdf.exists():
        os.makedirs(new_wmin_pdf)

    # write LHAPDF info file for new wmin pdf set
    write_new_lhapdf_info_file_from_previous_pdf(
        path_old_pdfset=wmin_basis_pdf,
        name_old_pdfset=wminpdfset,
        path_new_pdfset=new_wmin_pdf,
        name_new_pdfset=wmin_fit_name,
        num_members=ns_settings["n_posterior_samples"] + 1,
        description_set=f"Weight-minimized set using {wminpdfset} as basis",
        errortype=errortype,
    )

    # load replicas from basis set at all scales
    headers, grids = load_all_replicas(wminpdfset)
    replicas_df = rep_matrix(grids)

    wmin_parameters_sample = ultranest_fit.resampled_posterior

    # write replicas to new wmin pdf set
    write_wmin_combined_replicas(wmin_parameters_sample, replicas_df, new_wmin_pdf)

    # Generate central replica
    log.info(f"Generating central replica for {new_wmin_pdf}")
    generate_replica0(PDF(wmin_fit_name))

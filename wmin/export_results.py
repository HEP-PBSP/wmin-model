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

from validphys.lhio import load_all_replicas, rep_matrix, write_replica
from validphys.core import PDF

log = logging.getLogger(__name__)


def write_lhapdf_from_ultranest_result(
    wminpdfset,
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
    wminpdfset = PDF(wminpdfset)
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
    with open(wmin_basis_pdf / f"{wminpdfset}.info", "r") as in_stream, open(
        new_wmin_pdf / f"{wmin_fit_name}.info", "w"
    ) as out_stream:
        for l in in_stream.readlines():
            if l.find("SetDesc:") >= 0:
                out_stream.write(
                    f'SetDesc: "Weight-minimized set using {wminpdfset} as basis"\n'
                )
            elif l.find("NumMembers:") >= 0:
                out_stream.write(
                    f"NumMembers: {ns_settings["n_posterior_samples"] + 1}\n"
                )
            elif l.find("ErrorType: replicas") >= 0:
                out_stream.write(f"ErrorType: {errortype}\n")
            else:
                out_stream.write(l)

    headers, grids = load_all_replicas(wminpdfset)
    replicas_df = rep_matrix(grids)
    n_params = len(ultranest_fit.param_names)

    wmin_parameters_sample = ultranest_fit.resampled_posterior

    for i, wmin_weight in enumerate(wmin_parameters_sample):
        wmin_centr_rep, replica = (
            replicas_df.loc[:, [1]],
            replicas_df.loc[:, range(2, len(ultranest_fit.param_names) + 2)],
        )

        wm_replica = wmin_centr_rep.dot([1.0 - np.sum(wmin_weight)]) + replica.dot(
            wmin_weight
        )

        # for i, replica in tqdm(enumerate(result), total=len(weights)):
        wm_headers = f"PdfType: replica\nFormat: lhagrid1\nFromMCReplica: {i}\n"
        log.info(f"Writing replica {i + 1} to {new_wmin_pdf}")
        write_replica(i + 1, new_wmin_pdf, wm_headers.encode("UTF-8"), wm_replica)

    # Generate central replica

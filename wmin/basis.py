"""
wmin.basis.py

This module contains the functions that allow to construct a basis for the POD parametrisation.
"""

import logging
import os

import numpy as np
import tensorflow as tf
from colibri.constants import EXPORT_LABELS, LHAPDF_XGRID
from colibri.export_results import write_exportgrid
from n3fit.model_gen import _pdfNN_layer_generator, ReplicaSettings

from wmin.utils import (
    FLAV_INFO,
    arclength_outliers,
    arclength_pdfgrid,
)

log = logging.getLogger(__name__)


def n3fit_pdf_model(
    flav_info: list = FLAV_INFO,
    replica_range_settings: dict = {"min_replica": 1, "max_replica": 5},
    fitbasis: str = "EVOL",
    nodes: list = [25, 20, 8],
    activations: list = ["tanh", "tanh", "linear"],
    initializer_name: str = "glorot_normal",
    layer_type: str = "dense",
):
    """
    Wrapper function to generate a PDF model using the n3fit model generator.

    NOTE: in this function the n3fit model is always generated with the sum rules already imposed.
    However, for better stability, the sum rules are also imposed later-on in a more accurate way
    using a quadrature integration.
    """
    min_rep = replica_range_settings["min_replica"]
    max_rep = replica_range_settings["max_replica"]

    # Build one ReplicaSettings per replica
    replicas_settings = [
        ReplicaSettings(
            nodes=nodes,
            activations=activations,
            initializer=initializer_name,
            architecture=layer_type,
            seed=seed,
        )
        for seed in range(min_rep, max_rep + 1)
    ]

    pdf_model = _pdfNN_layer_generator(
        replicas_settings=replicas_settings,
        flav_info=flav_info,
        fitbasis=fitbasis,
        # leave impose_sumrule=None here so it defaults to "All"
        # and still produces xgrid_integration (x_in) as before
    )

    return pdf_model


def n3fit_pdf_grid(
    n3fit_pdf_model,
    filter_arclength_outliers: bool = True,
):
    """
    Returns the PDF grid for the n3fit model evaluated on the LHAPDF_XGRID.
    Also filters out the arclength outliers which can occurr when normalising the random
    PDF replicas for the sum rules.

    Parameters
    ----------
    n3fit_pdf_model: n3fit.model_gen._pdfNN_layer_generator
        The n3fit model to use.
    xgrid: array, default is LHAPDF_XGRID
        The xgrid to use.
    filter_arclength_outliers: bool, default is True
        Whether to filter out the arclength outliers from the PDF grid.

    Returns
    -------
    np.array
        The PDF grid for the n3fit model.
    """
    # TODO: write this using jax
    xgrid_in = tf.convert_to_tensor(np.array(LHAPDF_XGRID)[None, :, None])
    input = {"pdf_input": xgrid_in, "xgrid_integration": n3fit_pdf_model.x_in}

    pdf_grid = tf.squeeze(n3fit_pdf_model(input), axis=0)

    # shapes here are (nreplicas, nflavours, nx)
    pdf_array = np.array(tf.transpose(pdf_grid, perm=[0, 2, 1]))

    # filter from arclength outliers
    while filter_arclength_outliers:
        replicas_arclengths = arclength_pdfgrid(xgrid_in.numpy().squeeze(), pdf_array)
        # find outliers based on arclength interquartile range
        outliers = arclength_outliers(replicas_arclengths)

        log.info(f"Found {len(outliers)} arclength outliers in the PDF grid")

        # delete outliers from the grid
        pdf_array = np.delete(pdf_array, outliers, axis=0)

        # interrupt if no outliers are found
        if len(outliers) == 0:
            log.info("No more outliers found in the PDF grid")
            filter_arclength_outliers = False

    # Filter based on integrability of T3
    from colibri.constants import FLAVOUR_TO_ID_MAPPING
    t3_idx = FLAVOUR_TO_ID_MAPPING["T3"]
    t3_grid = pdf_array[:, t3_idx, :]
    tol = 5e-5
    integ_mask = abs(t3_grid[:,0]) <= tol
    pdf_array_clean = pdf_array[integ_mask, :, :]

    t8_idx = FLAVOUR_TO_ID_MAPPING["T8"]
    t8_grid = pdf_array_clean[:, t8_idx, :]
    integ_mask = abs(t8_grid[:,0]) <= tol
    pdf_array_clean = pdf_array_clean[integ_mask, :, :]

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(f"pdf_array shape = {pdf_array_clean.shape}")
    
    return pdf_array_clean


def get_X_matrix(pdf_grid: np.ndarray) -> tuple:
    """
    Convert and center (wrt to mean over replicas) the PDF grid
    to a 2D matrix suitable for singular value decomposition.

    Parameters
    ----------
    pdf_grid : numpy.ndarray, shape = (nreplicas, nflavours, nx)
        The PDF grid to be processed.

    Returns
    -------
    2-D tuple with:
        X : numpy.ndarray, shape = (nflavours * nx, nreplicas) = (ndata, nfeatures)
            The processed PDF grid.
        phi0 : numpy.ndarray, shape = (nflavours * nx, 1)
            The mean of the PDF grid over the replicas.
    """
    pdfgrid = pdf_grid.copy()
    # shape here is (Nreplicas, Nflavours x Nx)
    pdfgrid = pdfgrid.reshape(pdfgrid.shape[0], pdfgrid.shape[1] * pdfgrid.shape[2])

    # shape here is (Nflavours x Nx, Nreplicas)
    pdfgrid = pdfgrid.T

    # subtract the mean over the replicas (column-wise)
    phi0 = pdfgrid.mean(axis=1)[:, np.newaxis]
    pdfgrid -= phi0

    return pdfgrid, phi0


def pod_basis(n3fit_pdf_grid: np.ndarray, Neig: int) -> np.ndarray:
    """
    Performs a singular value decomposition (SVD) and Principal Component Analysis (PCA)
    on the PDF grid by returning the first Neig left singuar vectors.

    Parameters
    ----------
    n3fit_pdf_grid : numpy.ndarray, shape = (nreplicas, nflavours, nx)
        The PDF grid to be processed.
    Neig : int
        The number of eigenvectors to be returned.

    Returns
    -------
    U : numpy.ndarray, shape = (Neig, nflavours, nx)
        The first Neig left singular vectors of the PDF grid.
    """
    X, phi0 = get_X_matrix(n3fit_pdf_grid)

    # NOTE: only need left-singular matrix for POD
    U, S, _Vt = np.linalg.svd(X, full_matrices=False)

    # Select the first Neig singular vectors
    # NOTE: rescaling POD columns with singular values helps keeping the
    # wmin coefficents small during the fit.
    pod = (U @ np.diag(S))[:, :Neig]

    # Reshape U to (Neig, Nflavours, Nx)
    pod = (pod.T).reshape(Neig, n3fit_pdf_grid.shape[1], n3fit_pdf_grid.shape[2])
    phi0 = phi0.reshape(n3fit_pdf_grid.shape[1], n3fit_pdf_grid.shape[2])
    return pod, phi0


def write_pod_basis(
    pod_basis,
    output_path,
    Q=1.65,
    xgrid=LHAPDF_XGRID,
    export_labels=EXPORT_LABELS,
):
    """
    Writes the wmin basis at the parametrisation scale Q to the output_path.

    Parameters
    ----------
    pod_basis: tuple
        tuple containing the U matrix and phi0 vector.
    output_path: str
        The path to the output directory where the PDF grid will be written.
    Q: float, default is 1.65
        The scale at which to calculate the sum rules.
    xgrid: array, default is LHAPDF_XGRID
    export_labels: dict, default is EXPORT_LABELS

    """
    pod, phi0 = pod_basis
    basis = pod + phi0

    replicas_path = str(output_path) + "/replicas"
    if not os.path.exists(replicas_path):
        os.mkdir(replicas_path)

    fit_name = str(output_path).split("/")[-1]

    for i in range(basis.shape[0]):

        rep_path = replicas_path + f"/replica_{i+1}"

        if not os.path.exists(rep_path):
            os.mkdir(rep_path)

        grid_name = rep_path + "/" + fit_name

        if i == 0:
            # write the central member (phi0) of the basis to replica_1

            write_exportgrid(
                grid_for_writing=phi0,
                grid_name=grid_name,
                replica_index=i + 1,
                Q0=Q,
                xgrid=xgrid,
                export_labels=export_labels,
            )
        else:
            write_exportgrid(
                grid_for_writing=basis[i - 1],
                grid_name=grid_name,
                replica_index=i + 1,
                Q0=Q,
                xgrid=xgrid,
                export_labels=export_labels,
            )

    # TODO: how can we ensure that in the postfit of the evolution we don't by mistake also create another central member?
    log.info(
        f"Replicas written to {replicas_path}, with the central member at replica_1."
    )

    log.warning(
        "Note: this is a POD basis, so the central member is not the mean but always replica_1.\n"
        "After evolution, please run:\n"
        f"  python shift_lhadf_members.py {fit_name}/postfit/{fit_name}\n"
        "This will:\n"
        "  1. Remove the post-fit generated central member\n"
        "  2. Shift all others down by one index\n"
        "  3. Make replica_1 the new central member of the post-fit basis"
    )
    log.warning(
        "Reminder: decrement `NumMembers` by 1 in the LHAPDF .info file to reflect the removed member."
    )

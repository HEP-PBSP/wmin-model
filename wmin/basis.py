"""
wmin.basis.py

This module contains the functions that allow to construct a basis for the wmin parametrisation.
The main target used for the construction of the basis at this stage is the simultaneous 
satisfaction of the momentum, u-valence and d-valence sum rules.
"""

import logging
import os

import numpy as np
import tensorflow as tf
from colibri.constants import EXPORT_LABELS, LHAPDF_XGRID
from colibri.export_results import write_exportgrid
from n3fit.model_gen import pdfNN_layer_generator
from wmin.utils import FLAV_INFO_NNPDF40, arclength_outliers, arclength_pdfgrid

log = logging.getLogger(__name__)


def n3fit_pdf_model(
    flav_info: list = FLAV_INFO_NNPDF40,
    replica_range_settings: dict = {"min_replica": 1, "max_replica": 1000},
    impose_sumrule: bool = True,
    fitbasis: str = "EVOL",
    nodes: list = [25, 20, 8],
    activations: list = ["tanh", "tanh", "linear"],
    initializer_name: str = "glorot_normal",
    layer_type: str = "dense",
):
    """
    Wrapper function to generate a PDF model using the n3fit model generator.
    """
    pdf_model = pdfNN_layer_generator(
        nodes=nodes,
        activations=activations,
        initializer_name=initializer_name,
        layer_type=layer_type,
        seed=range(
            replica_range_settings["min_replica"],
            replica_range_settings["max_replica"] + 1,
        ),
        impose_sumrule=impose_sumrule,
        flav_info=flav_info,
        fitbasis=fitbasis,
        num_replicas=replica_range_settings["max_replica"]
        - replica_range_settings["min_replica"]
        + 1,
    )
    return pdf_model


def n3fit_pdf_grid(
    n3fit_pdf_model, xgrid=LHAPDF_XGRID, filter_arclength_outliers: bool = True
):
    """
    Returns the PDF grid for the n3fit model.
    Also filters out the arclength outliers which can occurr when normalising the random
    PDF replicas for the sum rules.

    Parameters
    ----------
    n3fit_pdf_model: n3fit.model_gen.pdfNN_layer_generator
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
    xgrid = tf.convert_to_tensor(np.array(xgrid)[None, :, None])
    input = {"pdf_input": xgrid, "xgrid_integration": n3fit_pdf_model.x_in}

    pdf_grid = tf.squeeze(n3fit_pdf_model(input), axis=0)

    pdf_array = np.array(tf.transpose(pdf_grid, perm=[0, 2, 1]))

    if filter_arclength_outliers:
        replicas_arclengths = arclength_pdfgrid(xgrid.numpy().squeeze(), pdf_array)
        # find outliers based on arclength interquartile range
        outliers = arclength_outliers(replicas_arclengths)

        log.info(f"Found {len(outliers)} arclength outliers in the PDF grid")

        # delete outliers from the grid
        pdf_array = np.delete(pdf_array, outliers, axis=0)

    return pdf_array


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
    U, _S, _Vt = np.linalg.svd(X, full_matrices=False)

    # Select the first Neig singular vectors
    U = U[:, :Neig]

    # Reshape U to (Neig, Nflavours, Nx)
    U = (U.T).reshape(Neig, n3fit_pdf_grid.shape[1], n3fit_pdf_grid.shape[2])
    phi0 = phi0.reshape(n3fit_pdf_grid.shape[1], n3fit_pdf_grid.shape[2])
    return U, phi0


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
    U_matrix, phi0 = pod_basis
    basis = U_matrix + phi0

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
            # write the central member of the basis
            rep_path_0 = replicas_path + f"/replica_0"

            if not os.path.exists(rep_path_0):
                os.mkdir(rep_path_0)

            grid_name_0 = rep_path_0 + "/" + fit_name

            write_exportgrid(
                grid_for_writing=phi0,
                grid_name=grid_name_0,
                replica_index=i,
                Q=Q,
                xgrid=xgrid,
                export_labels=export_labels,
            )

        write_exportgrid(
            grid_for_writing=basis[i],
            grid_name=grid_name,
            replica_index=i + 1,
            Q=Q,
            xgrid=xgrid,
            export_labels=export_labels,
        )

    # TODO: how can we ensure that in the postfit of the evolution we don't by mistake also create another central member?
    log.info(
        f"Replicas written to {replicas_path}, with the central member at replica_0."
    )
    log.info("Now you can evolve them with evolve_fit.")

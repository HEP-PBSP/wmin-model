"""
wmin.basis.py

This module contains the functions that allow to construct a basis for the POD parametrisation.
"""

import logging
import os

import numpy as np
from scipy.special import beta, gamma, betainc
import tensorflow as tf
from colibri.constants import EXPORT_LABELS, LHAPDF_XGRID, FLAVOUR_TO_ID_MAPPING
from colibri.export_results import write_exportgrid
from n3fit.model_gen import pdfNN_layer_generator

from wmin.utils import (
    FLAV_INFO,
    arclength_outliers,
    arclength_pdfgrid,
    sign_flip_selection,
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
    pdf_model = pdfNN_layer_generator(
        nodes=nodes,
        activations=activations,
        initializer_name=initializer_name,
        layer_type=layer_type,
        seed=range(
            replica_range_settings["min_replica"],
            replica_range_settings["max_replica"] + 1,
        ),
        impose_sumrule=True,  # sum-rules are also imposed later-on in a more accurate way.
        flav_info=flav_info,
        fitbasis=fitbasis,
        num_replicas=replica_range_settings["max_replica"]
        - replica_range_settings["min_replica"]
        + 1,
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
    # TODO: write this using jax
    xgrid_in = tf.convert_to_tensor(np.array(LHAPDF_XGRID)[None, :, None])
    input = {"pdf_input": xgrid_in, "xgrid_integration": n3fit_pdf_model.x_in}

    pdf_grid = tf.squeeze(n3fit_pdf_model(input), axis=0)

    # shapes here are (nreplicas, nflavours, nx)
    pdf_array = np.array(tf.transpose(pdf_grid, perm=[0, 2, 1]))

    # filter out replicas oscillating too much
    pdf_array = sign_flip_selection(pdf_array)

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

    return pdf_array


def XGRID():
    return np.array(LHAPDF_XGRID)


def gluon_singlet_prepro(
    XGRID: np.array, N_rep: int, seed: int = 1, sum_rules: bool = True
) -> np.array:
    """
    Returns an array of shape (N_rep, 2, N_x), where the first of the flavour indexes
    is the singlet and the second is the gluon.
    The N-replicas are randomly generated using uniform initialisation of the preprocessing
    exponents alpha and beta.
    When sum_rules is True, the sum rules are enforced.

    Parameters
    ----------
    XGRID : np.ndarray
        x grid.
    N_rep : int
        Number of replicas.
    seed : int, optional
        Seed for random number generator, by default 1.
    sum_rules : bool, optional
        Whether to enforce sum rules, by default True.

    Returns
    -------
    np.ndarray
        Array of gluon and singlet PDFs of shape (N_rep, 2, N_x).
    """
    sng_flav = FLAV_INFO[0]
    g_flav = FLAV_INFO[1]

    rng = np.random.default_rng(seed=np.random.randint(100000000))

    alphas_g = rng.uniform(
        low=g_flav["smallx"][0], high=g_flav["smallx"][1], size=N_rep
    )
    betas_g = rng.uniform(low=g_flav["largex"][0], high=g_flav["largex"][1], size=N_rep)

    alphas_sng = rng.uniform(
        low=sng_flav["smallx"][0], high=sng_flav["smallx"][1], size=N_rep
    )
    betas_sng = rng.uniform(
        low=sng_flav["largex"][0], high=sng_flav["largex"][1], size=N_rep
    )

    g_pdf = XGRID[:, None] ** (1 - alphas_g) * (1 - XGRID[:, None]) ** betas_g
    sng_pdf = XGRID[:, None] ** (1 - alphas_sng) * (1 - XGRID[:, None]) ** betas_sng

    # normalise PDF for sum rules (integral of x (g + sng))
    if sum_rules:
        norm = beta(2 - alphas_sng, 1 + betas_sng) + beta(2 - alphas_g, 1 + betas_g)
    else:
        norm = 1

    sng_pdf = sng_pdf / norm
    g_pdf = g_pdf / norm

    # stack pdfs together to form (N_rep, N_flav, N_x)
    pdf = np.stack([sng_pdf, g_pdf], axis=1)
    pdf = np.swapaxes(pdf, 0, 2)
    return pdf


def evolution_prepro(
    XGRID: np.array, flav: dict, N_rep: int, seed: int = 1, sum_rules: bool = True
) -> np.array:
    """
    For a given flavour, generate preprocessing PDF

    f_fl = A_fl x^(1 - alpha_fl) (1 - x)^beta_fl

    Parameters
    ----------
    XGRID : np.ndarray
        x grid.
    flav : dict
        Flavour information.
    N_rep : int
        Number of replicas.
    seed : int, optional
        Seed for random number generator, by default 1.
    sum_rules : bool, optional
        Whether to enforce sum rules, by default True.

    Returns
    -------
    np.ndarray
        Array of PDFs of shape (N_rep, N_x).
    """
    rng = np.random.default_rng(seed=np.random.randint(100000000))
    alphas = rng.uniform(low=flav["smallx"][0], high=flav["smallx"][1], size=N_rep)
    betas = rng.uniform(low=flav["largex"][0], high=flav["largex"][1], size=N_rep)

    pdf = XGRID[:, None] ** (-alphas) * (1 - XGRID[:, None]) ** betas

    # # normalise PDF for sum rules if needed
    if sum_rules:
        if flav["fl"] in ["v", "v3", "v8"]:
            norm = np.trapz(pdf, x=XGRID, axis=0)
            # norm = -betainc(1 - alphas, 1 + betas, XGRID[0]) + (gamma(1 - alphas) * gamma(1 + betas))/gamma(2 - alphas + betas)
            # norm = beta(1 - alphas, 1 + betas)
            # import IPython; IPython.embed()
            if flav["fl"] == "v" or flav["fl"] == "v8":
                pdf = 3 * pdf / norm
            elif flav["fl"] == "v3":
                pdf = pdf / norm
    xpdf = XGRID[:, None] * pdf
    return xpdf.T


def preprocessing_pdf_matrix(
    XGRID: np.array, N_rep: int, seed: int = 1, sum_rules: bool = True
) -> np.array:
    """
    Generate a matrix of preprocessing PDFs.
    The generated grid is supposed to be x f_k(x) and has shape (N_rep, N_flav, N_x).

    Parameters
    ----------
    XGRID : np.ndarray
        x grid.
    N_rep : int
        Number of replicas.
    seed : int, optional
        Seed for random number generator, by default 1.
    sum_rules : bool, optional
        Whether to enforce sum rules, by default True.

    Returns
    -------
    np.ndarray
        Array of PDFs of shape (N_rep, N_flav, N_x).
    """
    pdfs = []

    EV_FLAV_INFO_NNPDF40 = [
        flav for flav in FLAV_INFO if flav["fl"] not in ["sng", "g"]
    ]

    for flav in EV_FLAV_INFO_NNPDF40:
        pdf = evolution_prepro(XGRID, flav, N_rep, seed=seed, sum_rules=sum_rules)
        pdfs.append(pdf)

    sng_g = gluon_singlet_prepro(XGRID, N_rep, seed=seed, sum_rules=sum_rules)

    # reshape pdfs to (N_rep, N_flav, N_x)
    pdfs = np.array(pdfs)
    pdfs = np.swapaxes(pdfs, 0, 1)
    ic_pdfs_matrix = np.concatenate([sng_g, pdfs], axis=1)

    # Add all other flavours (tot=14) with the Intrinsic Charm relations
    # order is: "photon", "singlet", "g", "V", "V3", "V8", "V15", "V24", "V35", "T3", "T8", "T15", "T24", "T35"
    pdfs_matrix = np.zeros((N_rep, 14, len(XGRID)), dtype=np.float64)
    # photon is zero
    # singlet
    pdfs_matrix[:, FLAVOUR_TO_ID_MAPPING[r"\Sigma"], :] = ic_pdfs_matrix[:, 0, :]
    # gluon
    pdfs_matrix[:, FLAVOUR_TO_ID_MAPPING["g"], :] = ic_pdfs_matrix[:, 1, :]
    # V
    pdfs_matrix[:, FLAVOUR_TO_ID_MAPPING["V"], :] = ic_pdfs_matrix[:, 2, :]
    # V3
    pdfs_matrix[:, FLAVOUR_TO_ID_MAPPING["V3"], :] = ic_pdfs_matrix[:, 3, :]
    # V8
    pdfs_matrix[:, FLAVOUR_TO_ID_MAPPING["V8"], :] = ic_pdfs_matrix[:, 4, :]
    # V15 = V
    pdfs_matrix[:, FLAVOUR_TO_ID_MAPPING["V15"], :] = pdfs_matrix[
        :, FLAVOUR_TO_ID_MAPPING["V"], :
    ]
    # V24 = V
    pdfs_matrix[:, FLAVOUR_TO_ID_MAPPING["V24"], :] = pdfs_matrix[
        :, FLAVOUR_TO_ID_MAPPING["V"], :
    ]
    # V35 = V
    pdfs_matrix[:, FLAVOUR_TO_ID_MAPPING["V35"], :] = pdfs_matrix[
        :, FLAVOUR_TO_ID_MAPPING["V"], :
    ]
    # T3
    pdfs_matrix[:, FLAVOUR_TO_ID_MAPPING["T3"], :] = ic_pdfs_matrix[:, 5, :]
    # T8
    pdfs_matrix[:, FLAVOUR_TO_ID_MAPPING["T8"], :] = ic_pdfs_matrix[:, 6, :]
    # T15
    pdfs_matrix[:, FLAVOUR_TO_ID_MAPPING["T15"], :] = ic_pdfs_matrix[:, 7, :]
    # T24 = singlet
    pdfs_matrix[:, FLAVOUR_TO_ID_MAPPING["T24"], :] = pdfs_matrix[
        :, FLAVOUR_TO_ID_MAPPING[r"\Sigma"], :
    ]
    # T35 = singlet
    pdfs_matrix[:, FLAVOUR_TO_ID_MAPPING["T35"], :] = pdfs_matrix[
        :, FLAVOUR_TO_ID_MAPPING[r"\Sigma"], :
    ]

    # filter out replicas oscillating too much
    pdfs_matrix = sign_flip_selection(pdfs_matrix)

    # filter sum rules outliers
    from wmin.sr_normaliser import valence_sum_rules_outliers

    outlier_idxs = valence_sum_rules_outliers(pdfs_matrix, LHAPDF_XGRID)
    pdfs_matrix = np.delete(pdfs_matrix, outlier_idxs, axis=0)

    log.info(f"Identified {len(outlier_idxs)} sum rules outliers")
    log.info(f"New shape of pdf matrix is {pdfs_matrix.shape}")

    # filter from arclength outliers
    filter_arclength_outliers = True
    while filter_arclength_outliers:
        replicas_arclengths = arclength_pdfgrid(XGRID, pdfs_matrix)
        # find outliers based on arclength interquartile range
        outliers = arclength_outliers(replicas_arclengths)

        log.info(f"Found {len(outliers)} arclength outliers in the PDF grid")

        # delete outliers from the grid
        pdfs_matrix = np.delete(pdfs_matrix, outliers, axis=0)

        # interrupt if no outliers are found
        if len(outliers) == 0:
            log.info("No more outliers found in the PDF grid")
            filter_arclength_outliers = False

    return pdfs_matrix


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


def pod_basis(preprocessing_pdf_matrix: np.ndarray, Neig: int) -> np.ndarray:
    """
    Performs a singular value decomposition (SVD) and Principal Component Analysis (PCA)
    on the PDF grid by returning the first Neig left singuar vectors.

    Parameters
    ----------
    preprocessing_pdf_matrix : numpy.ndarray, shape = (nreplicas, nflavours, nx)
        The PDF grid to be processed.
    Neig : int
        The number of eigenvectors to be returned.

    Returns
    -------
    U : numpy.ndarray, shape = (Neig, nflavours, nx)
        The first Neig left singular vectors of the PDF grid.
    """
    X, phi0 = get_X_matrix(preprocessing_pdf_matrix)

    # NOTE: only need left-singular matrix for POD
    U, S, _Vt = np.linalg.svd(X, full_matrices=False)

    # Select the first Neig singular vectors
    # NOTE: rescaling POD columns with singular values helps keeping the
    # wmin coefficents small during the fit.
    pod = (U @ np.diag(S))[:, :Neig]

    # Reshape U to (Neig, Nflavours, Nx)
    pod = (pod.T).reshape(
        Neig, preprocessing_pdf_matrix.shape[1], preprocessing_pdf_matrix.shape[2]
    )
    phi0 = phi0.reshape(
        preprocessing_pdf_matrix.shape[1], preprocessing_pdf_matrix.shape[2]
    )
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
                Q=Q,
                xgrid=xgrid,
                export_labels=export_labels,
            )
        else:
            write_exportgrid(
                grid_for_writing=basis[i - 1],
                grid_name=grid_name,
                replica_index=i + 1,
                Q=Q,
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


def write_wmin_basis(
    wmin_basis_pdf_grid,
    output_path,
    Q=1.65,
    xgrid=LHAPDF_XGRID,
    export_labels=EXPORT_LABELS,
    replica_range_settings=None,
):
    """
    Writes the wmin basis at the parametrisation scale Q to the output_path.
    """

    replicas_path = str(output_path) + "/replicas"
    if not os.path.exists(replicas_path):
        os.mkdir(replicas_path)

    if replica_range_settings is not None:
        replica_range = range(
            replica_range_settings["min_replica"], replica_range_settings["max_replica"]
        )
        grid_index_range = range(1, wmin_basis_pdf_grid.shape[0])
    else:
        replica_range = range(1, wmin_basis_pdf_grid.shape[0])
        grid_index_range = range(1, wmin_basis_pdf_grid.shape[0])

    for replica_index, grid_index in zip(replica_range, grid_index_range):

        rep_path = replicas_path + f"/replica_{replica_index}"
        if not os.path.exists(rep_path):
            os.mkdir(rep_path)
        fit_name = str(output_path).split("/")[-1]

        grid_name = rep_path + "/" + fit_name

        write_exportgrid(
            grid_for_writing=wmin_basis_pdf_grid[grid_index],
            grid_name=grid_name,
            replica_index=replica_index,
            Q=Q,
            xgrid=xgrid,
            export_labels=export_labels,
        )

    log.info(f"Replicas written to {replicas_path}")
    log.info(
        "Now you can evolve them with evolve_fit and then compress them with mc2_hessian"
    )


def write_n3fit_basis(
    n3fit_pdf_grid,
    output_path,
    Q=1.65,
    xgrid=LHAPDF_XGRID,
    export_labels=EXPORT_LABELS,
    replica_range_settings=None,
):
    """
    Wrapper of write wmin basis for n3fit basis.
    """
    write_wmin_basis(
        n3fit_pdf_grid,
        output_path,
        Q=Q,
        xgrid=xgrid,
        export_labels=export_labels,
        replica_range_settings=replica_range_settings,
    )

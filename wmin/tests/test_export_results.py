from unittest.mock import patch, mock_open, MagicMock
import numpy as np
import os
import pathlib

# Assuming these functions are imported from the module
from wmin.export_results import (
    write_wmin_combined_replicas,
    write_new_lhapdf_info_file_from_previous_pdf,
    write_lhapdf_from_ultranest_result,
)


import numpy as np
import pandas as pd
import tempfile
import os
from unittest import mock
import pathlib


@mock.patch("wmin.export_results.write_replica")
@mock.patch("wmin.export_results.log.info")
def test_write_wmin_combined_replicas(mock_log_info, mock_write_replica):
    # Create mock wmin_parameters array
    wmin_parameters = np.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])

    # Create a mock replicas DataFrame
    data = {1: [0.5, 0.6], 2: [0.7, 0.8], 3: [0.9, 1.0], 4: [1.1, 1.2]}
    replicas_df = pd.DataFrame(data)

    # Temporary path for new_wmin_pdf
    with tempfile.TemporaryDirectory() as tmpdir:
        new_wmin_pdf = pathlib.Path(os.path.join(tmpdir, "wmin_pdf"))

        # Call the function being tested
        write_wmin_combined_replicas(wmin_parameters, replicas_df, new_wmin_pdf)

        # Check that log.info was called twice (since there are 2 wmin_parameter sets)
        assert mock_log_info.call_count == 2

        # Check that write_replica was called twice (once for each wmin_parameter set)
        assert mock_write_replica.call_count == 2

        # Check the arguments passed to write_replica
        call_args = mock_write_replica.call_args_list
        assert call_args[0][0][0] == 1  # Replica number 1
        assert call_args[1][0][0] == 2  # Replica number 2
        assert call_args[0][0][1] == new_wmin_pdf  # File path for new_wmin_pdf

        # Validate headers
        assert "PdfType: replica" in call_args[0][0][2].decode("utf-8")
        assert "FromMCReplica: 0" in call_args[0][0][2].decode("utf-8")
        assert "FromMCReplica: 1" in call_args[1][0][2].decode("utf-8")


# Mock the log.info call
@mock.patch("wmin.export_results.log.info")
def test_write_new_lhapdf_info_file_from_previous_pdf(mock_log_info):
    # Set up mock inputs
    name_old_pdfset = pathlib.Path("old_pdfset")
    name_new_pdfset = pathlib.Path("new_pdfset")
    num_members = 50
    description_set = "New Description"
    errortype = "custom_error"

    # Simulate old info file content
    old_info_content = """SetDesc: "Old Description"
NumMembers: 100
ErrorType: replicas
OtherInfo: "Some other details"
"""

    # Temporary directories to simulate old and new pdf sets
    with tempfile.TemporaryDirectory() as temp_old_dir, tempfile.TemporaryDirectory() as temp_new_dir:
        path_old_pdfset = pathlib.Path(temp_old_dir)
        path_new_pdfset = pathlib.Path(temp_new_dir)

        # Create a mock old info file
        old_info_file = pathlib.Path(
            os.path.join(path_old_pdfset, f"{name_old_pdfset}.info")
        )
        with open(old_info_file, "w") as f:
            f.write(old_info_content)

        # Call the function being tested
        write_new_lhapdf_info_file_from_previous_pdf(
            path_old_pdfset,
            name_old_pdfset,
            path_new_pdfset,
            name_new_pdfset,
            num_members,
            description_set,
            errortype,
        )

        # Verify that the new info file was created with correct content
        new_info_file = os.path.join(path_new_pdfset, f"{name_new_pdfset}.info")
        assert os.path.exists(new_info_file)

        with open(new_info_file, "r") as f:
            new_info_content = f.read()

        # Check that the updated fields are correct
        assert 'SetDesc: f"New Description"\n' in new_info_content
        assert f"NumMembers: {num_members}\n" in new_info_content
        assert f"ErrorType: {errortype}\n" in new_info_content
        assert (
            'OtherInfo: "Some other details"\n' in new_info_content
        )  # Unchanged content

        # Ensure that the log.info call was made with the correct message
        mock_log_info.assert_called_once_with(
            f"Info file written to {os.path.join(path_new_pdfset, f'{name_new_pdfset}.info')}"
        )


@patch("wmin.export_results.load_all_replicas")
@patch("wmin.export_results.rep_matrix")
@patch("wmin.export_results.write_replica")
@patch("wmin.export_results.generate_replica0")
@patch("os.makedirs")
@patch("pathlib.Path.exists", return_value=False)
@patch("wmin.export_results.lhapdf.paths")
@patch("wmin.export_results.PDF")
@patch("wmin.export_results.write_wmin_combined_replicas")
@patch("wmin.export_results.write_new_lhapdf_info_file_from_previous_pdf")
def test_write_lhapdf_from_ultranest_result(
    mock_write_info,
    mock_write_combined,
    mock_pdf,
    mock_lhapdf_paths,
    mock_path_exists,
    mock_makedirs,
    mock_generate_replica0,
    mock_write_replica,
    mock_rep_matrix,
    mock_load_all_replicas,
):
    # Setup mocks
    mock_pdf.return_value = "MockPDF"
    mock_lhapdf_paths.return_value = ["/path/to/lhapdf"]

    wmin_settings = {"wminpdfset": "MockPDF"}
    ultranest_fit = MagicMock()
    ultranest_fit.resampled_posterior = np.array(
        [[0.1, 0.2], [0.3, 0.4]]
    )  # Mock ultranest fit

    ns_settings = {"n_posterior_samples": 2}
    output_path = "/output/path"
    errortype = "replicas"

    mock_load_all_replicas.return_value = (["header1"], np.random.rand(5, 5))
    mock_rep_matrix.return_value = MagicMock()

    # Call the function
    write_lhapdf_from_ultranest_result(
        wmin_settings, ultranest_fit, ns_settings, output_path, errortype
    )

    mock_write_info.assert_called_once()  # Ensure the info file function was called
    mock_write_combined.assert_called_once()  # Ensure the combined replica function was called
    mock_generate_replica0.assert_called_once_with(
        "MockPDF"
    )  # Ensure central replica generation

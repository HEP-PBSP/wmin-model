from unittest import mock
from wmin.ultranest_fit import run_ultranest_fit


# Test when wmin_inherited_evolution is True
@mock.patch("wmin.ultranest_fit.write_lhapdf_from_ultranest_result")
def test_run_ultranest_fit_inherited_evolution(mock_write_lhapdf):
    # Set up mock inputs
    ultranest_fit = mock.Mock()  # Mock the ultranest_fit object
    output_path = mock.Mock()  # Mock the output_path (pathlib.PosixPath)
    pdf_model = mock.Mock()  # Mock the pdf_model
    wmin_settings = {
        "wmin_inherited_evolution": True
    }  # wmin_inherited_evolution is True
    ns_settings = {"some_ns_setting": 1}
    errortype = "replicas"

    # Call the function being tested
    run_ultranest_fit(
        ultranest_fit,
        output_path,
        pdf_model,
        wmin_settings,
        ns_settings,
        errortype,
    )

    # Verify that write_lhapdf_from_ultranest_result is called with correct arguments
    mock_write_lhapdf.assert_called_once_with(
        wmin_settings, ultranest_fit, ns_settings, output_path, errortype
    )


# Test when wmin_inherited_evolution is False
@mock.patch("colibri.ultranest_fit.run_ultranest_fit")
def test_run_ultranest_fit_no_inherited_evolution(mock_colibri_run):
    # Set up mock inputs
    ultranest_fit = mock.Mock()  # Mock the ultranest_fit object
    output_path = mock.Mock()  # Mock the output_path (pathlib.PosixPath)
    pdf_model = mock.Mock()  # Mock the pdf_model
    wmin_settings = {
        "wmin_inherited_evolution": False
    }  # wmin_inherited_evolution is False
    ns_settings = {"some_ns_setting": 1}
    errortype = "replicas"

    # Call the function being tested
    run_ultranest_fit(
        ultranest_fit,
        output_path,
        pdf_model,
        wmin_settings,
        ns_settings,
        errortype,
    )

    # Verify that colibri's run_ultranest_fit is called with correct arguments
    mock_colibri_run.assert_called_once_with(ultranest_fit, output_path, pdf_model)


# Test default behavior for errortype
@mock.patch("wmin.ultranest_fit.write_lhapdf_from_ultranest_result")
def test_run_ultranest_fit_default_errortype(mock_write_lhapdf):
    # Set up mock inputs
    ultranest_fit = mock.Mock()  # Mock the ultranest_fit object
    output_path = mock.Mock()  # Mock the output_path (pathlib.PosixPath)
    pdf_model = mock.Mock()  # Mock the pdf_model
    wmin_settings = {
        "wmin_inherited_evolution": True
    }  # wmin_inherited_evolution is True
    ns_settings = {"some_ns_setting": 1}

    # Call the function without providing an errortype (it should use the default "replicas")
    run_ultranest_fit(ultranest_fit, output_path, pdf_model, wmin_settings, ns_settings)

    # Verify that write_lhapdf_from_ultranest_result is called with the default "replicas" errortype
    mock_write_lhapdf.assert_called_once_with(
        wmin_settings, ultranest_fit, ns_settings, output_path, "replicas"
    )

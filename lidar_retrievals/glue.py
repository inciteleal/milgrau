"""
Function for pre-processing lidar signals - glue signal 

From original module provided by Ioannis Binietoglou on
https://gitlab.com/ioannis_binietoglou/lidar_molecular

# #############################################################################
#The MIT License (MIT)
#
#Copyright (c) 2015, Ioannis Binietoglou
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
# #############################################################################

"""

import numpy as np
from lidar_retrievals import fit_checks


def glue_signals_at_bins(lower_signal, upper_signal, min_bin, max_bin, c_lower, c_upper):
    """
    Glue two signals at a given bin range.

    The signal can be either a 1D array or a 2D array with dimensions (time, range).

    Both signals are assumed to have the same altitude grid. The final glued signal is calculated
    performing a linear fade-in/fade-out operation in the glue region.

    Parameters
    ----------
    lower_signal: array
       The low-range signal to be used. Can be either 1D or 2D with dimensions (time, range).
    upper_signal: array
       The high-range signal to be used. Can be either 1D or 2D with dimensions (time, range).
    min_bin: int
       The lower bin to perform the gluing
    max_bin: int
       The upper bin to perform the gluing
    c_lower: float
       Calibration constant of the lower signal. It will be equal to 1, if `use_upper_as_reference` argument
       is False.
    c_upper: float
       Calibration constant of the upper signal. It will be equal to 1, if `use_upper_as_reference` argument
       is True.

    Returns
    -------
    glued_signal: array
       The glued signal array, same size as lower_signal and upper_signal.
    """
    # Ensure that data are 2D-like
    if lower_signal.ndim == 1:
        lower_signal = lower_signal[np.newaxis, :]  # Force 2D
        upper_signal = upper_signal[np.newaxis, :]  # Force 2D
        axis_added = True
    else:
        axis_added = False

    gluing_length = max_bin - min_bin

    lower_weights = np.zeros_like(lower_signal)

    lower_weights[:, :min_bin] = 1
    lower_weights[:, min_bin:max_bin] = 1 - np.arange(gluing_length) / float(gluing_length)

    upper_weights = 1 - lower_weights

    # Calculate the glued signal
    glued_signal = c_lower * lower_weights * lower_signal + c_upper * upper_weights * upper_signal

    # Remove dummy axis, if added
    if axis_added:
        glued_signal = glued_signal[0, :]

    return glued_signal, c_lower, c_upper


def calculate_gluing_values(lower_gluing_region, upper_gluing_region, use_upper_as_reference):
    """
    Calculate the multiplicative calibration constants for gluing the two signals.

    Parameters
    ----------
    lower_gluing_region: array
       The low-range signal to be used. Can be either 1D or 2D with dimensions (time, range).
    upper_gluing_region: array
       The high-range signal to be used. Can be either 1D or 2D with dimensions (time, range).
    use_upper_as_reference: bool
       If True, the upper signal is used as reference. Else, the lower signal is used.

    Returns
    -------
    c_lower: float
       Calibration constant of the lower signal. It will be equal to 1, if `use_upper_as_reference` argument
       is False.
    c_upper: float
       Calibration constant of the upper signal. It will be equal to 1, if `use_upper_as_reference` argument
       is True.
    """
    lower_gluing_region = lower_gluing_region.ravel()  # Ensure we have an 1D array using ravel
    upper_gluing_region = upper_gluing_region.ravel()

    # Find their linear relationship, using least squares
    slope_zero_intercept, _, _, _ = np.linalg.lstsq(lower_gluing_region[:, np.newaxis], upper_gluing_region,rcond=None)

    # Set the calibration constants
    if use_upper_as_reference:
        c_upper = 1
        c_lower = slope_zero_intercept
    else:
        c_upper = 1 / slope_zero_intercept
        c_lower = 1

    return c_lower, c_upper


# def glue_signals_1d(lower_signal, upper_signal, window_length=200, correlation_threshold=0.95,
#                  intercept_threshold=0.5, gaussian_threshold=0.2, minmax_threshold=0.5,
#                  min_idx=None, max_idx=None, use_upper_as_reference=True):
    
def glue_signals_1d(lower_signal, upper_signal, window_length, correlation_threshold,
                 intercept_threshold, gaussian_threshold, minmax_threshold,
                 min_idx, max_idx, use_upper_as_reference=True):
    """
    Automatically glue two signals.

    Parameters
    ----------
    lower_signal: array
       The low-range signal to be used.
    upper_signal: array
       The high-range signal to be used.
    window_length: int
       The number of bins to be used for gluing
    correlation_threshold: float
       Threshold for the correlation coefficient
    intercept_threshold:
       Threshold for the linear fit intercept
    gaussian_threshold:
       Threshold for the Shapiro-Wilk p-value.
    minmax_threshold:
       Threshold for the min/max ratio
    min_idx, max_idx: int
       Minimum and maximum index to search for a gluing region.
    use_upper_as_reference: bool
       If True, the upper signal is used as reference. Else, the lower signal is used.

    Returns
    -------
    glued_signal: array
       The glued signal array, same size as lower_signal and upper_signal.
    gluing_center_idx: int
       Index choses to perform gluing.
    gluing_score: float
       The gluing score at the chosen point.
    c_lower, c_upper: floats
       Calibration constant of the lower and upper signal. One of them will be 1, depending on the
       value of `use_upper_as_reference` argument.
    """
    lower_signal_cut = lower_signal[min_idx:max_idx]
    upper_signal_cut = upper_signal[min_idx:max_idx]

    gluing_score = get_sliding_gluing_score(lower_signal_cut, upper_signal_cut, window_length, correlation_threshold,
                                            intercept_threshold, gaussian_threshold, minmax_threshold)

    gluing_center_idx = np.argmax(gluing_score) + min_idx  # Index of the original, uncut signals

    min_bin = int(gluing_center_idx - window_length // 2)
    max_bin = int(gluing_center_idx + window_length // 2)

    # Extract the gluing region
    # lower_gluing_region = lower_signal[:, min_bin:max_bin]
    # upper_gluing_region = upper_signal[:, min_bin:max_bin]
    
    lower_gluing_region = lower_signal[min_bin:max_bin]
    upper_gluing_region = upper_signal[min_bin:max_bin]

    # Calculate weights to fade-in/fade-out signals in gluing region.
    c_lower, c_upper = calculate_gluing_values(lower_gluing_region, upper_gluing_region, use_upper_as_reference)

    # Perform gluing
    glued_signal = glue_signals_at_bins(lower_signal, upper_signal, min_bin, max_bin, c_lower, c_upper)

    return glued_signal, gluing_center_idx, gluing_score[gluing_center_idx], c_lower, c_upper


def glue_signals_2d(lower_signal, upper_signal, correlation_threshold=0.95,
                 intercept_threshold=0.5, gaussian_threshold=0.2, minmax_threshold=0.5,
                 min_idx=None, max_idx=None, use_upper_as_reference=True):
    """
    Automatically glue two signals.

    Parameters
    ----------
    lower_signal: array
       The low-range signal to be used.
    upper_signal: array
       The high-range signal to be used.
    window_length: int
       The number of bins to be used for gluing
    correlation_threshold: float
       Threshold for the correlation coefficient
    intercept_threshold:
       Threshold for the linear fit intercept
    gaussian_threshold:
       Threshold for the Shapiro-Wilk p-value.
    minmax_threshold:
       Threshold for the min/max ratio
    min_idx, max_idx: int
       Minimum and maximum index to search for a gluing region.
    use_upper_as_reference: bool
       If True, the upper signal is used as reference. Else, the lower signal is used.

    Returns
    -------
    glued_signal: array
       The glued signal array, same size as lower_signal and upper_signal.
    gluing_center_idx: int
       Index choses to perform gluing.
    gluing_score: float
       The gluing score at the chosen point.
    c_lower, c_upper: floats
       Calibration constant of the lower and upper signal. One of them will be 1, depending on the
       value of `use_upper_as_reference` argument.
    """
    lower_signal_cut = lower_signal[:, min_idx:max_idx]
    upper_signal_cut = upper_signal[:, min_idx:max_idx]

    profile_number = lower_signal.shape[0]

    gluing_mask = []
    for profile_idx in range(profile_number):
        gluing_possible = check_gluing_possible(lower_signal_cut[profile_idx, :], upper_signal_cut[profile_idx, :],
                                            correlation_threshold, intercept_threshold, gaussian_threshold, minmax_threshold)
        gluing_mask.append(gluing_possible)

    gluing_mask = np.array(gluing_mask)

    # Extract the gluing region
    lower_gluing_region = lower_signal_cut[gluing_mask, :]
    upper_gluing_region = upper_signal_cut[gluing_mask, :]

    # Calculate weights to fade-in/fade-out signals in gluing region.
    c_lower, c_upper = calculate_gluing_values(lower_gluing_region, upper_gluing_region, use_upper_as_reference)

    # Perform gluing
    glued_signal = glue_signals_at_bins(lower_signal, upper_signal, min_idx, max_idx, c_lower, c_upper)

    return glued_signal, c_lower, c_upper


def get_sliding_gluing_score(lower_signal, upper_signal, window_length, correlation_threshold, intercept_threshold,
                             gaussian_threshold, minmax_threshold):
    """ Get gluing score.

    Parameters
    ----------
    lower_signal : array
       The low-range signal to be used.
    upper_signal : array
       The high-range signal to be used.
    window_length : int
       The number of bins to be used for gluing
    correlation_threshold : float
       Threshold for the correlation coefficient
    intercept_threshold : float
       Threshold for the linear fit intercept
    gaussian_threshold : float
       Threshold for the Shapiro-Wilk p-value.
    minmax_threshold : float
       Threshold for the min/max ratio.

    Returns
    -------
    gluing_score : masked array
       A score indicating regions were gluing is better. Regions were gluing is not possible are masked.
    """

    # Get values of various gluing tests
    intercept_values, correlation_values = fit_checks.sliding_check_linear_fit_intercept_and_correlation(lower_signal, upper_signal, window_length)
    
    gaussian_values = fit_checks.sliding_check_residuals_not_gaussian(lower_signal, upper_signal, window_length)
    minmax_ratio_values = fit_checks.sliding_check_min_max_ratio(lower_signal, upper_signal, window_length)

    # Find regions where all tests pass
    correlation_mask = correlation_values > correlation_threshold
    intercept_mask = intercept_values < intercept_threshold
    not_gaussian_mask = gaussian_values < gaussian_threshold
    minmax_ratio_large_mask = minmax_ratio_values > minmax_threshold

    gluing_possible = correlation_mask & intercept_mask & ~not_gaussian_mask & minmax_ratio_large_mask

    if not np.any(gluing_possible):
        raise RuntimeError("No suitable gluing regions found.")

    # Calculate a (arbitrary) cost function to deside which region is best
    intercept_scale_value = 40.
    intercept_values[intercept_values > intercept_scale_value] = intercept_scale_value
    intercept_score = 1 - intercept_values / intercept_scale_value

    gluing_score = correlation_values * intercept_score * minmax_ratio_values

    # Mask regions were gluing is not possible
    gluing_score = np.ma.masked_where(~gluing_possible, gluing_score)

    return gluing_score


def check_gluing_possible(lower_signal, upper_signal, correlation_threshold, intercept_threshold,
                             gaussian_threshold, minmax_threshold):
    """ Get gluing score.

    Parameters
    ----------
    lower_signal : array
       The low-range signal to be used.
    upper_signal : array
       The high-range signal to be used.
    correlation_threshold : float
       Threshold for the correlation coefficient
    intercept_threshold : float
       Threshold for the linear fit intercept
    gaussian_threshold : float
       Threshold for the Shapiro-Wilk p-value.
    minmax_threshold : float
       Threshold for the min/max ratio.

    Returns
    -------
    gluing_score : float or nan
       A score indicating regions were gluing is better. If gluing not possible, retunrs nan
    """

    # Get values of various gluing tests
    intercept_value, correlation_value = fit_checks.check_linear_fit_intercept_and_correlation(lower_signal, upper_signal)
    gaussian_value = fit_checks.check_residuals_not_gaussian(lower_signal, upper_signal)
    minmax_ratio_value = fit_checks.check_min_max_ratio(lower_signal, upper_signal)

    # Find regions where all tests pass
    correlation_mask = correlation_value > correlation_threshold
    intercept_mask = intercept_value < intercept_threshold
    not_gaussian_mask = gaussian_value < gaussian_threshold
    minmax_ratio_large_mask = minmax_ratio_value > minmax_threshold

    gluing_possible = correlation_mask & intercept_mask & ~not_gaussian_mask & minmax_ratio_large_mask

    return gluing_possible


def get_array_gluing_score(lower_signal, upper_signal, correlation_threshold, intercept_threshold,
                             gaussian_threshold, minmax_threshold):
    """ Get gluing score for 2D array.

    Parameters
    ----------
    lower_signal : array
       The low-range signal to be used.
    upper_signal : array
       The high-range signal to be used.
    correlation_threshold : float
       Threshold for the correlation coefficient
    intercept_threshold : float
       Threshold for the linear fit intercept
    gaussian_threshold : float
       Threshold for the Shapiro-Wilk p-value.
    minmax_threshold : float
       Threshold for the min/max ratio.

    Returns
    -------
    gluing_score : masked array
       A score indicating regions were gluing is better. Regions were gluing is not possible are masked.
    """

    # Get values of various gluing tests
    intercept_values, correlation_values = fit_checks.check_linear_fit_intercept_and_correlation(lower_signal, upper_signal)
    gaussian_values = fit_checks.check_residuals_not_gaussian(lower_signal, upper_signal)
    minmax_ratio_values = fit_checks.check_min_max_ratio(lower_signal, upper_signal)

    # Find regions where all tests pass
    correlation_mask = correlation_values > correlation_threshold
    intercept_mask = intercept_values < intercept_threshold
    not_gaussian_mask = gaussian_values < gaussian_threshold
    minmax_ratio_large_mask = minmax_ratio_values > minmax_threshold

    gluing_possible = correlation_mask & intercept_mask & ~not_gaussian_mask & minmax_ratio_large_mask

    if not np.any(gluing_possible):
        raise RuntimeError("No suitable gluing regions found.")

    # Calculate a (arbitrary) cost function to deside which region is best
    intercept_scale_value = 40.
    intercept_values[intercept_values > intercept_scale_value] = intercept_scale_value
    intercept_score = 1 - intercept_values / intercept_scale_value

    gluing_score = correlation_values * intercept_score * minmax_ratio_values

    # Mask regions were gluing is not possible
    gluing_score = np.ma.masked_where(~gluing_possible, gluing_score)

    return gluing_score
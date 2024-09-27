import numpy as np
from scipy import interpolate

#from .constants import k_b

c = 299792458

k_b = 1.38064852 * 10**-23  # Boltzmann constant in J/K

# Scattering parameters according to Freudenthaler V., 2015.
Cs = {308: 3.6506E-5,   # K/hPa/m
      351: 2.0934E-5,
      354.717: 2.0024E-5,
      355: 1.9957E-5,
      386.890: 1.3942E-5,
      400: 1.2109E-5,
      407.558: 1.1202E-5,
      510.6: 4.4221E-6,
      532: 3.7382E-6,
      532.075: 3.7361E-6,
      607.435: 2.1772E-6,
      710: 1.1561E-6,
      800: 7.1364E-7,
      1064: 2.2622E-7,
      1064.150: 2.2609E-7}


BsT = {308: 4.2886E-6,
       351: 2.4610E-6,
       354.717: 2.3542E-6,
       355: 2.3463E-6,
       400: 1.4242E-6,
       510.6: 5.2042E-7,
       532: 4.3997E-7,
       532.075: 4.3971E-7,
       710: 1.3611E-7,
       800: 8.4022E-8,
       1064: 2.6638E-8,
       1064.150: 2.6623E-8}

BsC = {308: 4.1678E-6,
       351: 2.3949E-6,
       354.717: 2.2912E-6,
       355: 2.2835E-6,
       400: 1.3872E-6,
       510.6: 5.0742E-7,
       532: 4.2903E-7,
       532.075: 4.2878E-7,
       710: 1.3280E-7,
       800: 8.1989E-8,
       1064: 2.5999E-8,
       1064.150: 2.5984E-8}

KbwT = {308: 1.01610,
        351: 1.01535,
        354.717: 1.01530,
        355: 1.01530,
        400: 1.01484,
        510.6: 1.01427,
        532: 1.01421,
        532.075: 1.01421,
        710: 1.01390,
        800: 1.01383,
        1064: 1.01371,
        1064.150: 1.01371}

KbwC = {308: 1.04554,
        351: 1.04338,
        354.717: 1.04324,
        355: 1.04323,
        400: 1.04191,
        510.6: 1.04026,
        532: 1.04007,
        532.075: 1.04007,
        710: 1.03919,
        800: 1.03897,
        1064: 1.03863,
        1064.150: 1.03863}

# Create interpolation function once, to avoid re-calculation (does it matter?)
f_ext = interpolate.interp1d(list(Cs.keys()), list(Cs.values()), kind='cubic')
f_bst = interpolate.interp1d(list(BsT.keys()), list(BsT.values()), kind='cubic')
f_bsc = interpolate.interp1d(list(BsC.keys()), list(BsC.values()), kind='cubic')


# Splines introduce arifacts due to limited input resolution
f_kbwt = interpolate.interp1d(list(KbwT.keys()), list(KbwT.values()), kind='linear')
f_kbwc = interpolate.interp1d(list(KbwC.keys()), list(KbwC.values()), kind='linear')


def standard_atmosphere(altitude):
    """
    Calculation of Temperature and Pressure in Standard Atmosphere.

    Parameters
    ----------
    altitude: float
       The altitude above sea level. (m)

    Returns
    -------
    pressure: float
       The atmospheric pressure. (N * m^-2 or Pa)
    temperature: float
       The atmospheric temperature. (K)
    density: float
       The air density. (kg * m^-3)

    References
    ----------
    http://home.anadolu.edu.tr/~mcavcar/common/ISAweb.pdf
    """
    # Temperature at sea level.
    temperature_sea_level = 288.15

    # Pressure at sea level.
    pressure_sea_level = 101325.

    # Dry air specific gas constant. (J * kg^-1 * K^-1)
    R = 287.058

    # Temperature calculation.
    temperature = temperature_sea_level - 6.5 * altitude / 1000.

    # Pressure calculation.
    pressure = pressure_sea_level * (1 - (0.0065 * altitude / temperature_sea_level)) ** 5.2561

    # Density calculation.
    density = pressure / (R * temperature)

    return pressure, temperature, density


def molecular_backscatter(wavelength, pressure, temperature, component='total'):
    """
    Molecular backscatter calculation.

    Parameters
    ----------
    wavelength : float
       The wavelength of the radiation in air. From 308 to 1064.15
    pressure : float
       The atmospheric pressure. (Pa)
    temperature : float
       The atmospheric temperature. (K)
    component : str
       One of 'total' or 'cabannes'.

    Returns
    -------
    beta_molecular: float
       The molecular backscatter coefficient. (m^-1 * sr^-1)

    References
    ----------
    Freudenthaler, V. Rayleigh scattering coefficients and linear depolarization
    ratios at several EARLINET lidar wavelengths. p.6-7 (2015)
    """
    if component not in ['total', 'cabannes']:
        raise ValueError("Molecular backscatter available only for 'total' or 'cabannes' component.")

    if component == 'total':
        bs_function = f_bst
    else:
        bs_function = f_bsc

    Bs = bs_function(wavelength)

    # Convert pressure to correct units for calculation. (Pa to hPa)
    pressure = pressure / 100.

    # Calculate the molecular backscatter coefficient.
    beta_molecular = Bs * pressure / temperature

    return beta_molecular


def molecular_lidar_ratio(wavelength, component='total'):
    """
    Molecular lidar ratio.

    Parameters
    ----------
    wavelength : float
       The wavelength of the radiation in air. From 308 to 1064.15
    component : str
       One of 'total' or 'cabannes'.

    Returns
    -------
    lidar_ratio_molecular : float
       The molecular backscatter coefficient. (m^-1 * sr^-1)

    References
    ----------
    Freudenthaler, V. Rayleigh scattering coefficients and linear depolarization
    ratios at several EARLINET lidar wavelengths. p.6-7 (2015)
    """
    if component not in ['total', 'cabannes']:
        raise ValueError("Molecular lidar ratio available only for 'total' or 'cabannes' component.")

    if component == 'total':
        k_function = f_kbwt
    else:
        k_function = f_kbwc

    Kbw = k_function(wavelength)

    lidar_ratio_molecular = 8 * np.pi / 3. * Kbw

    return lidar_ratio_molecular


def molecular_extinction(wavelength, pressure, temperature):
    """
    Molecular extinction calculation.

    Parameters
    ----------
    wavelength : float
       The wavelength of the radiation in air. From 308 to 1064.15
    pressure : float
       The atmospheric pressure. (Pa)
    temperature : float
       The atmospheric temperature. (K)

    Returns
    -------
    alpha_molecular: float
       The molecular extinction coefficient. (m^-1)

    References
    ----------
    Freudenthaler, V. Rayleigh scattering coefficients and linear depolarization
    ratios at several EARLINET lidar wavelengths. p.6-7 (2015)
    """
    cs = f_ext(wavelength)

    # Convert pressure to correct units for calculation. (Pa to hPa)
    pressure = pressure / 100.

    # Calculate the molecular backscatter coefficient.
    alpha_molecular = cs * pressure / temperature

    return alpha_molecular


def number_density_at_pt(pressure, temperature):
    """ Calculate the number density for a given temperature and pressure.

    This method does not take into account the compressibility of air.

    Parameters
    ----------
    pressure: float or array
       Pressure in Pa
    temperature: float or array
       Temperature in K

    Returns
    -------
    n: array or array
       Number density of the atmosphere [m-3]
    """
    #p_pa = pressure * 100.  # Pressure in pascal

    n = pressure / (temperature * k_b)

    return n

#!/usr/bin/env python
# coding: utf-8

import struct
from pathlib import Path
from typing import Dict,Tuple,List,BinaryIO,Optional

import numpy as np
import xarray as xr
from netCDF4 import Dataset
from scipy.interpolate import griddata


# Paths

if not((Path.cwd()/'../Data').exists()):
    raise Exception('Folder Data and structure must exist')

PATH = Path.cwd()/'../Data/NOAA/Atmospherical_Conditions'
PRESSURE_PATH = Path.cwd()/'../Data/NOAA/Pressure_Conditions'
FORECASTED_PATH = Path.cwd()/'../Data/SPEEDY'
INTERPOLATIONS_PATH = Path.cwd()/'../Data/Interpolations'

INTERPOLATION_VARIABLES = 4

####---------------

YEAR = '2020'

FILES = [f'uwnd.{YEAR}.nc', f'vwnd.{YEAR}.nc',
         f'air.{YEAR}.nc', f'rhum.{YEAR}.nc']

FORECASTED_FILES = [k.name for k in FORECASTED_PATH.rglob('*.grd')]


# Temporal setting
DATE = '2020-07-01'
TIME = '00:00:00'
# Time is HH:MM:SS in 24-hours format
DATETIME = DATE + 'T' + TIME
FILENAME = DATE.replace('-', '') + TIME[:2]

####---------------

IS_CONVERTION_REQUIRED = False
SAVE_AS_GRD = False

####---------------


# Default pressure leves
PRESSURE_LEVELS_VALUES = [925, 850, 700, 500, 300, 200, 100]

# Grids definition
# SPEEDY

SPEEDY_LON = 96
SPEEDY_LAT = 48
SPEEDY_LVL = 7

X_SPEEDY_LON = np.linspace(0, 360-3.75, 96)
Y_SPEEDY_LAT = np.array("-87.159 -83.479 -79.777 -76.070 -72.362 -68.652 -64.942 -61.232 -57.521 -53.810 -50.099 -46.389 -42.678 -38.967 -35.256 -31.545 -27.833 -24.122 -20.411 -16.700 -12.989 -9.278 -5.567 -1.856 1.856 5.567 9.278 12.989 16.700 20.411 24.122 27.833 31.545 35.256 38.967 42.678 46.389 50.099 53.810 57.521 61.232 64.942 68.652 72.362 76.070 79.777 83.479 87.159".split(" "))
Y_SPEEDY_LAT = Y_SPEEDY_LAT.astype(np.float32)


# NOAA: latitude goes from North To South
NOAA_LON = 144
NOAA_LAT = 73
NOAA_LVL = 7

X_NOAA_LON = np.linspace(0, 360-2.5, 144)
Y_NOAA_LAT = np.linspace(90, -90, 73)
X_grid_noaa, Y_grid_noaa = np.meshgrid(X_NOAA_LON, Y_NOAA_LAT)

## Eviroment functions
def set_default_paths(**kargs: str) -> None:
    """
    Function that recieves any specific path and changes it.

    Parameters
    ----------
    noaa_path : str
        New PATH for NOAA files.
    pressure_path : str
        New PATH for NOAA pressure files.
    intepolation_path : str
        New PATH for Interpolation files.
    speedy_path : str
        New PATH for SPEEDY model files.

    Returns
    -------
    None.

    """
    n_args = len(kargs)
    valid_keys = ['noaa_path','pressure_path','intepolation_path','speedy_path']

    if(n_args == 0 or n_args > 4):
        print('Number of arguments invalid')
        return

    if not(all(key in valid_keys for key in kargs)):
        print('There is an invalid argument. Please check the valid parameters')
        return

    if(valid_keys[0] in kargs):
        PATH = kargs[valid_keys[0]]
        print(f'The new Path for NOAA files is {PATH}')

    if(valid_keys[1] in kargs):
        PATH = kargs[valid_keys[1]]
        print(f'The new Path for NOAA pressure files is {PATH}')

    if(valid_keys[2] in kargs):
        PATH = kargs[valid_keys[2]]
        print(f'The new Path for Interpolation files is {PATH}')

    if(valid_keys[3] in kargs):
        PATH = kargs[valid_keys[3]]
        print(f'The new Path for SPEEDY files is {PATH}')

def set_default_dates(year: str,month: str,day: str,hour: str) -> None:
    """
    Function to set default date to perform analysis

    Parameters
    ----------
    year : str
        Desired year to analyze.
    month : str
        Desired month to analyze.
    day : str
        Desired day to analyze.
    hour : str
        Desired hout to analyze.
    """
    YEAR = year

    FILES = [f'uwnd.{YEAR}.nc', f'vwnd.{YEAR}.nc',
             f'air.{YEAR}.nc', f'rhum.{YEAR}.nc']

    DATE = f'{year}-{mont}-{day}'
    TIME = f'{hour}:00:00'

    DATETIME = DATE + 'T' + TIME
    FILENAME = DATE.replace('-', '') + TIME[:2]

def set_default_conversion_parameters(convertion: bool, grd: bool) -> None:
    """
    Sets parameters to performe relative humidity to specific convertion
    and to save as grd

    Parameters
    ----------
    convertion : bool
        Defines if is required to convert relative humidity to specific
    grd : bool
        Defines if it is necesary to save it as grd
    """
    IS_CONVERTION_REQUIRED = convertion
    SAVE_AS_GRD = grd

## NOAA FUNCTIONS
def read_data(variable: str, file: str) -> xr.Dataset:
    '''
    Reads nc files from the NOAA.

    Parameters
    ----------
    variable : str
        Desired variable to get information.
    file : str
        name of the file that contains specific variable.

    Returns
    -------
    variable_array: xr.Dataset
        n-dimensional xarray with the choosen pressure levels for the given variable
    '''

    try:
        variable_path: Path = PATH/file
        variable_array: xr.Dataset = xr.open_dataset(variable_path)[variable].sel(
            level=PRESSURE_LEVELS_VALUES,
            time=DATETIME)
    except Exception as e:
        raise Exception("File not defined in specified path." +
         " Please check if PATH is correctly setted or if file exists."+
         " Remember the NOAA naming convention {variable}.{year}.nc")

    return variable_array

def relative2specific(T: float, RH: float, p:float) ->float:
    '''
    Converts relative humidity to specific humidity

    Parameters
    ----------
    T : float
        Temperature in K.
    RH : float
        Relative humidity in percentage [0,100].
    p : float
        Preassure in mbar.

    Returns
    -------
    shum: float
        Conversion value of specific humidity

    '''
    T -= 273.15
    p *= 100
    RH /= 100
    e_s: float = 611.21*np.exp((18.687-T/234.5)*(T/(T+257.14)))
    e: float = e_s*RH
    w: float = 287.058/461.5*e/(p-e)
    shum: float = w/(w+1)
    return shum

def convert_relative_humidity_to_specific(atmospherical_variables: Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
    """
    Takes the Numpy array contaning variables, and performs the convertion of
    relative humidity to specific making use of our function. The resulting
    array does not contain rhum, as it is eliminated.

    Parameters
    ----------
    atmospherical_variables : Dict[str, np.ndarray]
        Numpy array containing atmospherical variables values, including, relative humidity.

    Returns
    -------
    atmospherical_variables : Dict[str, np.ndarray]
        Numpy array containing atmospherical variables values with the convertion of humidity.
    """
    specific_humidity_values: np.ndarray =  np.zeros((SPEEDY_LVL,SPEEDY_LAT,SPEEDY_LON))

    for index,pressure_value in enumerate(PRESSURE_LEVELS_VALUES):
        RH_data: np.ndarray = atmospherical_variables['rhum'][index,:,:].copy()
        air_T_data: np.ndarray = atmospherical_variables['air'][index,:,:].copy()
        specific_humidity_values[index,:,:] = np.vectorize(relative2specific)(air_T_data,
                                                                          RH_data,
                                                                          pressure_value)

    atmospherical_variables['shum'] = specific_humidity_values
    atmospherical_variables.pop('rhum',None);
    return atmospherical_variables
    # pop deletes the values in the dict which are contained in the Key. If no key is found, None is returned.
    # This is done, as the rhum variable is no longer needed in the process.
    # To check funcionality, please create a copy of dict at this point.

def read_noaa_variables() -> Optional[Dict[str,np.ndarray]]:
    """
    Reads information from NetCDF files coming from NOAA repositories,
    and stored in the specified folder path.

    Returns
    -------
    atmospherical_variables : Dict[str, np.ndarray]
        contains atmospherical data, by levels, over all the grid.
    """
    atmospherical_variables: Dict[str,np.ndarray] = dict()

    for file in FILES:
        variable: str = file.split(".")[0]
        variable_values_by_level: np.ndarray = np.zeros((NOAA_LVL, NOAA_LAT, NOAA_LON))

        variable_array = read_data(variable, file)

        for index_pressure_level, pressure_level in enumerate(PRESSURE_LEVELS_VALUES):
            variable_values_by_level[index_pressure_level, :,
                                     :] = variable_array.sel(level=pressure_level).values

        atmospherical_variables[variable] = variable_values_by_level

    if (IS_CONVERTION_REQUIRED):
        atmospherical_variables = convert_relative_humidity_to_specific(atmospherical_variables)

    return atmospherical_variables

def format_and_save_noaa_dataset(atmospherical_variables: np.ndarray) -> None:
    """
    Formats numpy array having the information in order to store it as
    an Dataset. Information is processed to have the scheme required
    by xarray. Check http://xarray.pydata.org/en/stable/data-structures.html
    Order of data is first Level, second latitude and finally longitute

    Parameters
    ----------
    atmospherical_variables : np.ndarray
        Numpy variable containing atmospherical information.
    """
    atmospherical_variables_to_netcdf: Dict[str,Tuple(Tuple(str,str,str),np.ndarray)] = dict()

    for key, value in atmospherical_variables.items():
        atmospherical_variables_to_netcdf[key] = (("level", "lat", "lon"), value)

    atmospherical_dataset: xr.Dataset = xr.Dataset(
        atmospherical_variables_to_netcdf,
        coords={
            "level": PRESSURE_LEVELS_VALUES,
            "lat": Y_NOAA_LAT,
            "lon": X_NOAA_LON,
        },
        attrs={
            'long_name': '6-Hourly Sample',
            'Levels': NOAA_LVL,
            'dataset': 'NCEP/DOE AMIP-II Reanalysis (Reanalysis-2)',
            'level_desc': 'Surface',
            'statistic': 'Individual Obs',
        },
    )

    atmospherical_dataset.to_netcdf(INTERPOLATIONS_PATH/("NOAA-" + FILENAME + "-atmospherical_dataset.nc"))

    if(SAVE_AS_GRD):
        save_as_grd(atmospherical_variables)

def save_as_grd(atmospherical_variables: np.ndarray) -> None:
    """
    Stores dataset as an grd file, to serve as SPEEDY input. Saves it in
    the Interpolations folder.

    Parameters
    ----------
    atmospherical_variables : np.ndarray
        Numpy variable containing atmospherical information.
    """
    result_list: List = list()
    for variable in atmospherical_variables.values():
        result_list.extend(variable.ravel().tolist())

    fout: BinaryIO  = open(INTERPOLATIONS_PATH/(FILENAME+'.grd'), 'wb')
    for i in result_list:
        fout.write(struct.pack('>f',i))
    fout.close()

## SPEEDY Functions
def read_grd(filename: str) -> Dict[str,np.ndarray]:
    """
    Reads data from SPEEDY comming in grd file.

    Parameters
    ----------
    filename : str
        String containing the name of the file to be loaded.

    Returns
    -------
    variables_SPEEDY: Dict[str, np.ndarray]
        Structure containing variables loaded from a grd file, in SPEEDY
        data format.
    """

    infile: BinaryIO = open(FORECASTED_PATH/filename, "rb")
    data: np.ndarray = np.fromfile(infile, '>f4')
    l: int = 0
    U: np.ndarray = np.empty([SPEEDY_LVL, SPEEDY_LAT, SPEEDY_LON])
    for k in range(SPEEDY_LVL):
        for j in range(SPEEDY_LAT):
            for i in range(SPEEDY_LON):
                U[k, j, i] = data[l]
                l = l+1

    V: np.ndarray = np.empty([SPEEDY_LVL, SPEEDY_LAT, SPEEDY_LON])
    for k in range(SPEEDY_LVL):
        for j in range(SPEEDY_LAT):
            for i in range(SPEEDY_LON):
                V[k, j, i] = data[l]
                l = l+1

    T: np.ndarray = np.empty([SPEEDY_LVL, SPEEDY_LAT, SPEEDY_LON])
    for k in range(SPEEDY_LVL):
        for j in range(SPEEDY_LAT):
            for i in range(SPEEDY_LON):
                T[k, j, i] = data[l]
                l = l+1

    SH: np.ndarray = np.empty([SPEEDY_LVL, SPEEDY_LAT, SPEEDY_LON])
    for k in range(SPEEDY_LVL):
        for j in range(SPEEDY_LAT):
            for i in range(SPEEDY_LON):
                SH[k, j, i] = data[l]
                l = l+1

    P: np.ndarray = np.empty([SPEEDY_LAT, SPEEDY_LON])
    for j in range(SPEEDY_LAT):  # range(nlat-1,-1,-1): #range(nlat):
        for i in range(SPEEDY_LON):
            P[j, i] = data[l]
            l = l+1

    variables_SPEEDY: Dict[str,np.ndarray] = dict()
    variables_SPEEDY['uwnd'] = U
    variables_SPEEDY['vwnd'] = V
    variables_SPEEDY['temperature'] = T
    variables_SPEEDY['shum'] = SH
    variables_SPEEDY['pres'] = P

    return variables_SPEEDY

def save_speedy_as_nc(variables_SPEEDY: Dict[str,np.ndarray]) -> None:

    SPEEDY_atmospherical_variables_to_netcdf: Dict[str,Tuple(Tuple(str,str,str),np.ndarray)] = dict()
    SPEEDY_pressure_to_netcdf: Dict[str,Tuple(Tuple(str,str),np.ndarray)]  = dict()

    pressure: np.ndarray = variables_SPEEDY.pop('pres', None)

    for key, value in variables_SPEEDY.items():
        SPEEDY_atmospherical_variables_to_netcdf[key] = (
            ("level", "lat", "lon"), value)

    speedy_atmospherical_dataset: xr.Dataset = xr.Dataset(
        SPEEDY_atmospherical_variables_to_netcdf,
        coords={
            "level": PRESSURE_LEVELS_VALUES,
            "lat": Y_SPEEDY_LAT,
            "lon": X_SPEEDY_LON,
        },
        attrs={
            'long_name': '6-Hourly Sample',
            'Levels': 7,
            'dataset': 'NCEP/DOE AMIP-II Reanalysis (Reanalysis-2)',
            'level_desc': 'Surface',
            'statistic': 'Individual Obs',
        },
    )

    SPEEDY_pressure_to_netcdf['pres'] = (("lat", "lon"), pressure)
    SPEEDY_pressure_dataset: xr.Dataset = xr.Dataset(
        SPEEDY_pressure_to_netcdf,
        coords={
            "lat": Y_SPEEDY_LAT,
            "lon": X_SPEEDY_LON,
        },
        attrs={
            'long_name': '6-Hourly Pressure at Surface',
            'Levels': 1,
            'units': 'Pascals',
            'precision': -1,
            'GRIB_id': 1,
            'GRIB_name': 'PRES',
            'var_desc': 'Pressure',
            'dataset': 'NCEP/DOE AMIP-II Reanalysis (Reanalysis-2)',
            'level_desc': 'Surface',
            'statistic': 'Individual Obs',
            'parent_stat': 'Other',
            'standard_name': 'pressure',
        },
    )

    speedy_atmospherical_dataset.to_netcdf(
        INTERPOLATIONS_PATH/('SPEEDY-'+FILENAME + "-atmospherical_dataset.nc"))
    SPEEDY_pressure_dataset.to_netcdf(
        INTERPOLATIONS_PATH/('SPEEDY-'+FILENAME + "-pressure_dataset.nc"))


def execute_noaa_convertion() -> None:
    """
    Funtion that perfomes all related to NOAA saving
    """
    atmospherical_variables: xr.Dataset = read_noaa_variables()
    format_and_save_noaa_dataset(atmospherical_variables)

def do_all() -> None:
    """
    Function that performs both NOAA and SPEEDY saving.
    """
    execute_noaa_convertion()
    save_speedy_as_nc()

if __name__ == '__main__':
    do_all()
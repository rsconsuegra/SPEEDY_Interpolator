# SPEEDY_Interpolator
Here lies an interpolator to SPEEDY grided format, particullary, comming from the NOAA repository.

# TO DO
- Further description required
- Use xarray in order to efficiently iterate over netCDF file
- Dinamically generate name of interpoalted data to be executed in SPEEDY model
- Refactor Interpolation and Error calculator Notebook
- Grammar check

# To Consider
- Make use of Docker to garantee reproductivity

The folder Structure must be:

SPEEDY-Interpoler/
├── .ipynb_checkpoints/
│   └── Untitled-checkpoint.ipynb
├── Data/
│   ├── NOAA/
│   │   ├── Atmospherical_Conditions/
│   │   │   └── **NOAA NETCDF Files of Atmospherical Conditions**
│   │   └── Pressure_Conditions/
│   │       └── **NOAA NETCDF Files relative of Pressure on Surface information**
│   └── SPEEDY/
│       └── **SPEEDY model .grd Files**
└──Notebooks/
    └── **Required Jupyter Notebooks to perform interpolation and error estimation**
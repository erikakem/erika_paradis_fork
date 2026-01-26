import numpy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import re 
import xarray 
import os
import matplotlib.pyplot as plt 
from scores.probability import crps_for_ensemble

from data.datamodule import Era5DataModule

# Upgrade: Just Give just version number
ensemble_members = [
                    '/home/erk002/store5/erika_paradis_fork/erika_paradis_fork/forecasts/test_member_version41_1deg_1',
                    '/home/erk002/store5/erika_paradis_fork/erika_paradis_fork/forecasts/test_member_version41_1deg_2',
                    '/home/erk002/store5/erika_paradis_fork/erika_paradis_fork/forecasts/test_member_version41_1deg_3',
                    '/home/erk002/store5/erika_paradis_fork/erika_paradis_fork/forecasts/test_member_version41_1deg_4',
                    '/home/erk002/store5/erika_paradis_fork/erika_paradis_fork/forecasts/test_member_version41_1deg_5',
                    ]
datasets = [xarray.open_mfdataset(
    member,
    chunks={"time": 1},
    engine="zarr",
) for member in ensemble_members] 



# Combine into one dataset with a new 'member' dimension
ensemble_ds = xarray.concat(datasets, dim="member")
#ensemble_ds = ensemble_ds.assign_coords(member=list(range(len(datasets))))


# Get the verifying observation: ERA5 measurement of the actual weather for the day in question (must be in the past)

f = ensemble_ds["2m_temperature"]

# Load verifying observation (ERA5 observation)
obs_ds = xarray.open_mfdataset(
    "/home/erk002/store5/erika_paradis_fork/erika_paradis_fork/forecasts/test_member_version41_1deg_1_ERA5",
    chunks={"time": 1},
    engine="zarr",
)

obs = obs_ds["2m_temperature"]

# --------------------
# Align obs and forecasts
# --------------------
obs, f = xarray.align(obs, f, join="inner")
# print(obs.time.values)


fcrps = crps_for_ensemble(f,obs,ensemble_member_dim='member', method='fair')
crps = crps_for_ensemble(f,obs,ensemble_member_dim='member', method='ecdf')

print("crps value:", float(crps.compute()))
print("fcrps value:", float(fcrps.compute()))



    

    


    

        
        
    

   


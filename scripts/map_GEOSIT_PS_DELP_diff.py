# Map of the difference between two values of surface pressure (PS) from GEOS-IT at an 
# example time (2018-01-01 00:00:00 UTC):
# (1) PS provided as a 2D variable in GEOS-IT test files
# (2) The surface pressure calculated by summing the difference between pressure levels (DELP) across
#     all model levels in 3D files, plus the pressure at model top (0.01 hPa)

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

GEOSIT_data_dir = '/data/GEOS-IT_test/2018_test/001/'

nc_ASM_I1 = Dataset(GEOSIT_data_dir+'GEOS.it.asm.asm_inst_1hr_glo_L576x361_slv.GEOS5271.'+\
                    '2018-01-01T0000.V01.nc4')

nc_ASM_I3 = Dataset(GEOSIT_data_dir+'GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5271.'+\
                    '2018-01-01T0000.V01.nc4')

PS = np.array(nc_ASM_I1.variables['PS'][0][:])

DELP = np.array(nc_ASM_I3.variables['DELP'][0][:][:][:])
DELP_sum = 1 + np.sum(DELP, axis=0)

PS_diff = PS - DELP_sum

fig = plt.figure()

m = Basemap(projection='robin',lon_0=0,resolution='c')
m.drawcoastlines()
m.drawparallels(np.arange(-90.,120.,30.))
m.drawmeridians(np.arange(0.,360.,60.))

lons = np.array(nc_ASM_I1.variables['lon'])
lats = np.array(nc_ASM_I1.variables['lat'])

lons_mesh, lats_mesh = np.meshgrid(lons, lats)
x, y = m(lons_mesh, lats_mesh)

diff_clevs = np.arange(-15,16,1)
diff_cmap = plt.get_cmap('RdBu_r')

PS_diff_fill = plt.contourf(x, y, PS_diff, diff_clevs, cmap=diff_cmap, extend='both')

cbar = plt.colorbar(PS_diff_fill, extend='both')
cbar.ax.set_title('PS - DELP sum (Pa)')

#plt.show()

plt.savefig('map_GEOSIT_PS_DELP_diff.png', dpi=300, bbox_inches='tight')

plt.close('all')

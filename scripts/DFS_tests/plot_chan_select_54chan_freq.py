import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import netCDF4
import copy
import TIRS_channel_cmap

#choose case to plot
# 0= Arctic ocean, 1=Greenland, 2=Antarctica, 3=Tropics
case = 0

srffile = '/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.10.4_360_2021-03-28_PCRTM_grid.nc'

if case == 0:
    vfilename = '/data/users/nnn/DFS_analysis/channel_select/ERA5_DFSselect_pwv_SaMod01_Ocean_at1000xt8_faster.h5'
if case == 1:
    vfilename = '/data/users/nnn/DFS_analysis/channel_select/ERA5_DFSselect_pwv_SaMod01_Greenland_at1000xt8_faster.h5'
if case == 2:
    vfilename = '/data/users/nnn/DFS_analysis/channel_select/ERA5_DFSselect_pwv_SaMod01_Antarctica_at1000xt8_faster.h5'
if case == 3:
    vfilename = '/data/users/nnn/DFS_analysis/channel_select/ERA5_DFSselect_pwv_SaMod01_Tropics_at1000xt8_faster.h5'


vdat = {}
with h5py.File(vfilename, 'r') as h:
    for dataset in h:
        vdat[dataset] = h[dataset][:]

chan_order = vdat['chan_order'][:].astype(np.int)
chan_total_dfs = vdat['chan_total_dfs'][:]
pwv = vdat['pwv'][:]


new_shape = (np.prod(chan_order.shape[0:2]),chan_order.shape[2])
chan_order_2D = np.reshape(chan_order, new_shape)
chan_total_dfs_2D = np.reshape(chan_total_dfs, new_shape)

new_shape2 = (np.prod(chan_order.shape[0:2]))
pwv_1D = np.reshape(pwv, new_shape2)


nc2 = netCDF4.Dataset(srffile,'r')
srf_norm = nc2['srf_normed'][:].astype(np.float32)
srf = nc2['srf'][:].astype(np.float32)
wn1 = nc2['channel_wavenum1'][:,0]
wn2 = nc2['channel_wavenum2'][:,0]
wn = 0.5 * (wn1 + wn2)
wnr = np.r_[wn1, wn2[-1]]
wn_test = nc2['wavenum'][:]

wl1 = nc2['channel_wavelen1'][:,0]
wl2 = nc2['channel_wavelen2'][:,0]
wl = 0.5 * (wl1 + wl2)
wlr = np.r_[wl1, wl2[-1]]

#raise ValueError()

wl_shape = wl.shape[0]
wlnum_good = 54
totalnum = chan_order_2D.shape[0]

data = np.zeros([wlnum_good,wl_shape])

for i in range(wlnum_good):
    print('i = '+str(i))
    for j in range(1,len(wl)+1):
        #check to see how often a channel(j) occurs for this position(i)
        curr_wl = chan_order_2D[:,i] == j
        
        #find the fraction it occurs in the position
        curr_wl_pcent = sum(curr_wl)/totalnum
        
        data[i,j-1] = curr_wl_pcent


 # definitions for the axes
left, width = 0.12, 0.85
bottom, height = 0.300, 0.625
spacing = 0.005
cb_bottom = 0.09
cb_height = 0.03

rect_scatter = [left, bottom, width, height]
rect_cb = [left, cb_bottom, width, cb_height]



fignum =2
fig = plt.figure(fignum, figsize=(15.1, 6.2))
fig.clf()

ax = fig.add_axes(rect_scatter)
ax_cb = fig.add_axes(rect_cb)


labels = (np.array(range(1,55)))

wl_use = np.zeros([wl_shape])
data_cum = data.cumsum(axis=1)

cmapin = TIRS_channel_cmap.get_listed_cmap()
# colors are set according to a 0-indexed channel ID number,
# where channel #0 is the direct, undispersed beam.
# channel #0 isn't in the SRF file so we start at 1.
category_colors = cmapin(np.arange(1, 64))
width = 1.0


for k in range(len(wl)):
    if sum(data[:,k]) >= 0.99:
            #print(wl[k])
            wl_use[k] = copy.deepcopy(wl[k])
            heights = data[:, k]
            starts = data_cum[:, k] - heights
            rects = ax.bar(labels,heights,width,bottom=starts,
                           color=category_colors[k,:])

if case == 0:
    titlename = 'Arctic Ocean'
if case == 1:
    titlename = 'Greenland'
if case == 2:
    titlename = 'Antarctica'
if case == 3:
    titlename = 'Tropics'

ax.set_xlabel('Channel Rank',fontsize=18)
ax.set_ylabel('Fractional Occurrence',fontsize=18)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14 )
ax.set_title(titlename,fontsize=20)
ax.set_xlim(0, 55)

# creating ScalarMappable for color bar: this needs to be the
# channel colormap split out from channel number 1 - 63.
sm = plt.cm.ScalarMappable(
    norm=mpl.colors.Normalize(vmin=0, vmax=64), 
    cmap=TIRS_channel_cmap.get_listed_cmap())
# not sure why we need to do this (not needed in mpl 3.3)
sm.set_array([])

filter_block_edge_channels = np.array([4, 7, 10, 16, 19, 34, 37, 63])
filter_block_wl_ranges = [wl[k-1] for k in filter_block_edge_channels]
filter_block_wl_tickl = ['{:2.1f}'.format(w) for w in filter_block_wl_ranges]

# note we add 0.49 to the tick location so that the tick looks like
# it is pointing to the middle of that color in the color bar.
cb = plt.colorbar(
    sm, cax=ax_cb, orientation='horizontal',
    ticks = filter_block_edge_channels + 0.49)

cb.ax.set_xticklabels(filter_block_wl_tickl)

cbax2 = cb.ax.twiny()
cbax2.set_xlim(0,64)
# here, add 0.49 so that it gets rounded down for display.
# a bit hacky, but works.
cbax2.set_xticks(filter_block_edge_channels+0.49)
cbax2.xaxis.set_major_formatter(
    mpl.ticker.StrMethodFormatter('{x:2.0f}'))

cb.set_label('Wavelength [$\mu$m]',size=12)
cb.ax.tick_params(labelsize=12)
cbax2.tick_params(labelsize=12)


plt.show()


savedat = 0
if savedat == 1:
    outfile = '/data/users/nnn/DFS_analysis/channel_select/Tropics_chanrank_data.h5'
    h = h5py.File(outfile,'w')
    h['chanrank_data'] =  data
    h['center_wavelength'] = wl

    h.close()


raise ValueError()



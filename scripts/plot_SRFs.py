"""
simple visualizations of the SRF files - does a simple numbering/labeling
of each SRF profile while plotting the whole array of them across 0 - 60 um.
(annotated_SRF_plot)

NEDR_plot shows the difference between v0.03 and v0.04, to see the impact
of the slit width.

"""
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import os.path

from mpl_toolkits.axes_grid1 import ImageGrid


def visualize_v10v11_NEDR(fignum=30):

    # index 125 is 275 K.
    T_idx = 125

    with netCDF4.Dataset('../data/PREFIRE_SRF_v0.10.4_360_2021-03-28.nc','r') as nc:
        nedr10 = nc['NEDR'][:]
        drdT10 = nc['drad_dT'][T_idx]
        nedt10 = nedr10/drdT10

    with netCDF4.Dataset('../data/PREFIRE_TIRS1_SRF_v11_2022-12-10.nc','r') as nc:
        nedr11_t1 = nc['NEDR'][:]
        mask11_t1 = nc['detector_mask'][:]
        drdT11_t1 = nc['drad_dT'][T_idx]
        nedt11_t1 = nedr11_t1/drdT11_t1

    with netCDF4.Dataset('../data/PREFIRE_TIRS2_SRF_v11_2022-12-10.nc','r') as nc:
        nedr11_t2 = nc['NEDR'][:]
        mask11_t2 = nc['detector_mask'][:]
        drdT11_t2 = nc['drad_dT'][T_idx]
        nedt11_t2 = nedr11_t2/drdT11_t2

    pooled_nedr11 = np.concatenate([nedr11_t1, nedr11_t2], axis=1)
    pooled_mask11 = np.concatenate([mask11_t1, mask11_t2], axis=1)

    median_good_nedr11 = np.zeros(63)
    for c in range(63):
        msk_c = pooled_mask11[c,:] == 0
        if np.any(msk_c):
            median_good_nedr11[c] = np.median(pooled_nedr11[c,msk_c])

    scaled_nedr10 = nedr10 / median_good_nedr11[:,np.newaxis]
    scaled_nedr11_t1 = nedr11_t1 / median_good_nedr11[:,np.newaxis]
    scaled_nedr11_t2 = nedr11_t2 / median_good_nedr11[:,np.newaxis]

    prec10 = np.zeros(8)
    prec11_t1 = np.zeros(8)
    prec11_t2 = np.zeros(8)
    for s in range(8):
        nedr = nedr10[:,s]
        msk = nedr<1e9
        prec10[s] = np.sqrt(np.sum(1/nedr[msk]**2))
        nedr = nedr11_t1[:,s]
        msk = mask11_t1[:,s] < 2
        prec11_t1[s] = np.sqrt(np.sum(1/nedr[msk]**2))
        nedr = nedr11_t2[:,s]
        msk = mask11_t2[:,s] < 2
        prec11_t2[s] = np.sqrt(np.sum(1/nedr[msk]**2))


    nedt10_ma = np.ma.masked_where(scaled_nedr10 > 1e9, nedt10)
    nedt11_t1_ma = np.ma.masked_where(mask11_t1 >= 2, nedt11_t1)
    nedt11_t2_ma = np.ma.masked_where(mask11_t2 >= 2, nedt11_t2)

    scaled_nedr10_ma = np.ma.masked_where(scaled_nedr10 > 1e9, scaled_nedr10)
    scaled_nedr11_t1_ma = np.ma.masked_where(mask11_t1 >= 2, scaled_nedr11_t1)
    scaled_nedr11_t2_ma = np.ma.masked_where(mask11_t2 >= 2, scaled_nedr11_t2)

    #### Figure 1
    fig = plt.figure(fignum, figsize=(8,5))
    fig.clf()
    grid = ImageGrid(fig, 111, nrows_ncols=(3,1), axes_pad=0.3,
                     cbar_mode='single', cbar_size='1%')
    pm = grid[0].pcolormesh(np.log2(scaled_nedr10_ma.T), vmin=-1.6, vmax=1.6, cmap='coolwarm')
    pm = grid[1].pcolormesh(np.log2(scaled_nedr11_t1_ma.T), vmin=-1.6, vmax=1.6, cmap='coolwarm')
    pm = grid[2].pcolormesh(np.log2(scaled_nedr11_t2_ma.T), vmin=-1.6, vmax=1.6, cmap='coolwarm')

    grid[1].set_ylabel('Scene number')
    grid[0].set_title('SRF v10')
    grid[1].set_title('SRF v11 TIRS 1')
    grid[2].set_title('SRF v11 TIRS 2')
    grid[2].set_xlabel('Channel number')
    grid.cbar_axes[0].colorbar(pm)
    grid.cbar_axes[0].set_title('Log2\n NEDR ratio')

    fig.tight_layout()
    fig.savefig('plots/SRF_v10v11_comparison_01_NEDRratio_image.png')

    #### Figure 2
    fig = plt.figure(fignum+1, figsize=(8,5))
    fig.clf()

    ax = fig.add_subplot(111)
    obj, = ax.plot(scaled_nedr10_ma[:,0], '-o', label='scaled NEDR v10')
    obj, = ax.plot(scaled_nedr11_t1_ma[:,0], '-s', label='scaled NEDR v11 TIRS 1', alpha=0.3)
    ax.plot(scaled_nedr11_t1_ma[:,1:], '-s', color=obj.get_color(), alpha=0.5)
    obj, = ax.plot(scaled_nedr11_t2_ma[:,0], '-v', label='scaled NEDR v11 TIRS 2', alpha=0.3)
    ax.plot(scaled_nedr11_t2_ma[:,1:], '-v', color=obj.get_color(), alpha=0.5)
    ax.legend(loc='upper right')

    ax.set_ylim(0.34, 3.15)
    ax.set_xlabel('Channel number')
    ax.set_ylabel('NEDR ratio')
    ax.grid(1)

    fig.tight_layout()
    fig.savefig('plots/SRF_v10v11_comparison_02_NEDRratio_lineplots.png')

    #### Figure 3
    fig = plt.figure(fignum+2, figsize=(8,5))
    fig.clf()

    ax = fig.add_subplot(111)
    obj, = ax.semilogy(nedt10_ma[:,0], '-o', label='NEDT v10')
    obj, = ax.semilogy(nedt11_t1_ma[:,0], '-s', label='NEDT v11 TIRS 1', alpha=0.3)
    ax.plot(nedt11_t1_ma[:,1:], '-s', color=obj.get_color(), alpha=0.5)
    obj, = ax.semilogy(nedt11_t2_ma[:,0], '-v', label='NEDT v11 TIRS 2', alpha=0.3)
    ax.plot(nedt11_t2_ma[:,1:], '-v', color=obj.get_color(), alpha=0.5)
    ax.legend(loc='lower right')

    ax.set_ylim(0.05, 30.0)
    ax.set_xlabel('Channel number')
    ax.set_ylabel('NEDT at 275 K')
    ax.grid(1)

    fig.tight_layout()
    fig.savefig('plots/SRF_v10v11_comparison_02_NEDT_lineplots.png')

    #### Figure 4
    fig = plt.figure(fignum+3)
    fig.clf()

    ax = fig.add_subplot(111)

    scene_number = np.arange(1,9)
    ax.plot(scene_number, prec10, '-o', label='SRFv10.4 precision')
    ax.plot(scene_number, prec11_t1, '-s', label='SRFv11 TIRS1 precision')
    ax.plot(scene_number, prec11_t2, '-v', label='SRFv11 TIRS2 precision')
    ax.legend(loc='best')
    ax.set_xlabel('Scene number')
    ax.set_ylabel('Precision total [1 / (W/(m^2 sr um))]')
    ax.grid(1)

    fig.tight_layout()
    fig.savefig('plots/SRF_v10v11_comparison_03_precision_totals.png')


def visualize_pooled_NEDR_and_masks():
    """
    one use function to make some visualization plots of the NEDR
    and the resulting masks

    see process_srf.assess_pooled_NEDR()
    """

    with netCDF4.Dataset('../data/PREFIRE_TIRS1_SRF_v0.11_FPA_mask.nc','r') as nc:
        NEDR1 = nc['NEDR'][:]
        scaled_NEDR1 = nc['scaled_NEDR'][:]
        mask1 = nc['mask'][:]
    with netCDF4.Dataset('../data/PREFIRE_TIRS2_SRF_v0.11_FPA_mask.nc','r') as nc:
        NEDR2 = nc['NEDR'][:]
        scaled_NEDR2 = nc['scaled_NEDR'][:]
        mask2 = nc['mask'][:]

    log_n_bins = np.logspace(0, 4.5, 136)
    zoom_n_bins = np.linspace(1,4,61)

    fig = plt.figure(20)
    fig.clf()

    fig, ax = plt.subplots(2,1,num=20)

    ax[0].hist(scaled_NEDR1.flatten(), log_n_bins)
    ax[0].set_xlim(0.5,20000)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xticklabels('')
    ax[0].set_title('TIRS1 scaled NEDR')

    ax[1].hist(scaled_NEDR2.flatten(), log_n_bins)
    ax[1].set_xlim(0.5,20000)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_title('TIRS2 scaled NEDR')

    fig.savefig('plots/SRF_v11_scaledNEDR_hist.png')

    fig.clf()
    fig, ax = plt.subplots(2,1,num=20)

    ax[0].hist(scaled_NEDR1.flatten(), zoom_n_bins)
    ax[0].set_xticklabels('')
    ax[0].set_title('TIRS1 scaled NEDR')

    ax[1].hist(scaled_NEDR2.flatten(), zoom_n_bins)
    ax[1].set_title('TIRS2 scaled NEDR')

    fig.savefig('plots/SRF_v11_scaledNEDR_hist_zoom.png')

    # in log10 space:
    # the range of NEDR is: -2.5 -> 2.75
    # scaled NEDR is:        0.0 -> 4.25

    fig.clf()
    fig, ax = plt.subplots(2,1,num=20)

    pm = ax[0].pcolormesh(np.log10(NEDR1.T), vmin=-2.5, vmax=2.75,
                          cmap='gnuplot')
    ax[0].set_xticklabels('')
    ax[0].set_title('TIRS1 NEDR')
    ax[0].set_ylabel('Scene number')
    cb = plt.colorbar(pm, ax=ax[0])
    cb.ax.set_title('Log10 NEDR')

    pm = ax[1].pcolormesh(np.log10(NEDR2.T), vmin=-2.5, vmax=2.75,
                          cmap='gnuplot')
    ax[1].set_title('TIRS2 NEDR')
    ax[1].set_ylabel('Scene number')
    ax[1].set_xlabel('Channel number')
    cb = plt.colorbar(pm, ax=ax[1])
    cb.ax.set_title('Log10 NEDR')

    fig.savefig('plots/SRF_v11_NEDR_2Dimg.png')

    fig.clf()
    fig, ax = plt.subplots(2,1,num=20)

    pm = ax[0].pcolormesh(np.log10(scaled_NEDR1.T), vmin=0, vmax=4.25, cmap='gnuplot')
    ax[0].set_xticklabels('')
    ax[0].set_title('TIRS1 scaled NEDR')
    ax[0].set_ylabel('Scene number')
    cb = plt.colorbar(pm, ax=ax[0])
    cb.ax.set_title('Log10 NEDR')

    pm = ax[1].pcolormesh(np.log10(scaled_NEDR2.T), vmin=0, vmax=4.25, cmap='gnuplot')
    ax[1].set_title('TIRS2 scaled NEDR')
    ax[1].set_ylabel('Scene number')
    ax[1].set_xlabel('Channel number')
    cb = plt.colorbar(pm, ax=ax[1])
    cb.ax.set_title('Log10 NEDR')

    fig.savefig('plots/SRF_v11_scaledNEDR_2Dimg.png')

    fig.clf()
    fig, ax = plt.subplots(2,1,num=20)

    pm = ax[0].pcolormesh(np.log10(scaled_NEDR1.T), vmin=0, vmax=2, cmap='seismic')
    ax[0].set_xticklabels('')
    ax[0].set_title('TIRS1 scaled NEDR')
    ax[0].set_ylabel('Scene number')
    cb = plt.colorbar(pm, ax=ax[0])
    cb.ax.set_title('Log10 NEDR')

    pm = ax[1].pcolormesh(np.log10(scaled_NEDR2.T), vmin=0, vmax=2, cmap='seismic')
    ax[1].set_title('TIRS2 scaled NEDR')
    ax[1].set_ylabel('Scene number')
    ax[1].set_xlabel('Channel number')
    cb = plt.colorbar(pm, ax=ax[1])
    cb.ax.set_title('Log10 NEDR')

    fig.savefig('plots/SRF_v11_scaledNEDR_zoom_2Dimg.png')

    fig.clf()
    fig, ax = plt.subplots(2,1,num=20)

    pm = ax[0].pcolormesh(mask1.T, cmap='gnuplot')
    ax[0].set_xticklabels('')
    ax[0].set_title('TIRS1 detector mask')
    ax[0].set_ylabel('Scene number')
    cb = plt.colorbar(pm, ax=ax[0])

    pm = ax[1].pcolormesh(mask2.T, cmap='gnuplot')
    ax[1].set_title('TIRS2 detector mask')
    ax[1].set_ylabel('Scene number')
    ax[1].set_xlabel('Channel number')
    cb = plt.colorbar(pm, ax=ax[1])

    fig.savefig('plots/SRF_v11_detectormasks_2Dimg.png')




def annotated_SRF_plot(ncfile, ax=None, fignum=20):

    with netCDF4.Dataset(ncfile, 'r') as nc:
        wavelen = nc['wavelen'][:]
        srf = nc['srf'][:]
        w1 = nc['channel_wavelen1'][:]
        w2 = nc['channel_wavelen2'][:]
    w = 0.5*(w1+w2)

    if ax is None:
        fig = plt.figure(fignum, figsize=(16,4))
        fig.clf()
        ax = fig.add_subplot(111)

    for n in range(srf.shape[1]):
        obj, = ax.plot(wavelen, srf[:,n])
        ax.text(w[n], 0.235, str(n+1), ha='center', color=obj.get_color())
        ax.plot([w[n],w[n]], [0, 0.225], '--', color=obj.get_color(), lw=1)

    ax.set_ylim(0,0.25)
    ax.set_xlim(0,58)
    ax.set_ylabel('SRF Amplitude')
    ax.set_xlabel('Wavelength [um]')

    ax.set_title('Data from ' + ncfile.split('/')[-1])

    if ax is None:
        fig.tight_layout()


def NEDR_plot(file1, file2=None, fignum=21):

    with netCDF4.Dataset(file1, 'r') as nc:
        channel_wavelen1 = nc['channel_wavelen1'][:]
        channel_wavelen2 = nc['channel_wavelen2'][:]
        NEDR_1 = nc['NEDR'][:]
    w1 = 0.5*(channel_wavelen1 + channel_wavelen2)

    if file2 is not None:
        with netCDF4.Dataset(file2, 'r') as nc:
            channel_wavelen1 = nc['channel_wavelen1'][:]
            channel_wavelen2 = nc['channel_wavelen2'][:]
            NEDR_2 = nc['NEDR'][:]
        w2 = 0.5*(channel_wavelen1 + channel_wavelen2)

    fig = plt.figure(fignum)
    fig.clf()
    ax = fig.add_subplot(111)

    if w1.ndim == 2:
        w1 = w1[:,0]
    if NEDR_1.ndim == 2:
        NEDR_1 = NEDR_1[:,0]

    if file2 is not None:
        if w2.ndim == 2:
            w2 = w2[:,0]
        if NEDR_2.ndim == 2:
            NEDR_2 = NEDR_2[:,0]
    print(NEDR_1.shape, NEDR_2.shape)

    ax.plot(w1, NEDR_1, 'o', label=os.path.split(file1)[1])
    if file2 is not None:
        ax.plot(w2, NEDR_2, 'o', label=os.path.split(file2)[1])
    ax.grid(1)
    ax.set_ylim(0,0.045)
    ax.legend(loc='upper right')
    ax.set_xlabel('Wavelength [um]')
    ax.set_ylabel('NEDR [W/m^2/sr/um]')

    if file2 is None:
        ax.legend([os.path.split(file1)[1]], 
                  loc=(0.2,1.0))
    else:
        ax.legend([os.path.split(file1)[1],
                   os.path.split(file2)[1]], 
                  loc=(0.2,1.0), ncol=2)



def grouped_SRF_plot(srf_file1, srf_file2 = None, fignum=20):

    with netCDF4.Dataset(srf_file1, 'r') as nc:
        wavelen = nc['wavelen'][:]
        srf = nc['srf'][:]
        w1 = nc['channel_wavelen1'][:]
        w2 = nc['channel_wavelen2'][:]
        filt_num = nc['filter_number'][:]

    if srf.ndim == 3:
        srf = srf[:,:,0]

    if srf_file2:
        with netCDF4.Dataset(srf_file2, 'r') as nc:
            wavelen2 = nc['wavelen'][:]
            srf2 = nc['srf'][:]
        if srf2.ndim == 3:
            srf2 = srf2[:,:,0]
    else:
        srf2 = None

    w = 0.5*(w1+w2)

    fig = plt.figure(fignum, figsize=(15,10))
    fig.clf()
    ax = [fig.add_subplot(4,1,p) for p in range(1,5)]

    for n in range(4):
        for k in range(1+n, 63, 4):
            ax[n].plot(wavelen, srf[:,k], color='b')
            if srf2 is not None:
                ax[n].plot(wavelen2, srf2[:,k], color='g')
            if filt_num[k] >= 2:
                ax[n].text(w[k], 0.025, str(k+1), ha='center', color='b')
        if n == 0:
            if srf2 is None:
                ax[n].legend([os.path.split(srf_file1)[1]], 
                             loc=(0.2,1.0))
            else:
                ax[n].legend([os.path.split(srf_file1)[1],
                              os.path.split(srf_file2)[1]], 
                             loc=(0.2,1.0), ncol=2)
        ax[n].grid(1)
        ax[n].set_xlim(2, 58)
        if n < 3:
            ax[n].set_xticklabels('')
        else:
            ax[n].set_xlabel('Wavelength [um]')

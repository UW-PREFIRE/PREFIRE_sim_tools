import h5py
import numpy as np
import matplotlib.pyplot as plt

import PREFIRE_plot_helpers

def scatterhex_hist_DFS_PWV(x, y, logx=False, **kw):

    fig, objs = PREFIRE_plot_helpers.scatterhex_hist(
        x, y, logx=logx, cmap='cool', gridsize=40, **kw)

    ax, ax_histx, ax_histy, ax_cb, cb = objs

    if logx:
        #xlims = -1.5, 0.75
        xlims = -1.5,0.89
        xlabel = 'log10 PWV [cm]'
    else:
        #xlims = -0.1, 3.9
        xlims  = -0.1,7.2
        xlabel = 'PWV [cm]'
#    ylims = 0.3, 7.2
    ylims = 0.3, 8.2

    ax.set(xlabel=xlabel,ylabel='DFS')
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    return fig

def batch_make_plots():

    #filenames = ['/data/users/nnn/ERA5/h5/ERA5_DFS_diag_pwv_Ocean.h5',
    #             '/data/users/nnn/ERA5/h5/ERA5_DFS_diag_pwv_Greenland.h5',
    #             '/data/users/nnn/ERA5/h5/ERA5_DFS_diag_pwv_Antarctica.h5']
    filenames = ['/data/users/nnn/DFS_analysis/ERA5_DFS_diag_pwv_srf9p2_Ocean.h5',
                 '/data/users/nnn/DFS_analysis/ERA5_DFS_diag_pwv_srf9p2_Greenland.h5',
                 '/data/users/nnn/DFS_analysis/ERA5_DFS_diag_pwv_srf9p2_Antarctica.h5',
                 '/data/users/nnn/DFS_analysis/ERA5_DFS_diag_pwv_srf9p2_Tropics.h5']
    labels = [f.split('_')[-1][:-3] for f in filenames]
    print(labels)

    for f, label in zip(filenames, labels):

        with h5py.File(f, 'r') as h:
            dt = h['dfs_temp'][:].flatten()
            dq = h['dfs_q'][:].flatten()
            dtq = h['dfs_tq'][:].flatten()
            pwv = h['pwv_interp'][:].flatten()

        fig = scatterhex_hist_DFS_PWV(
            pwv, dq, logx=True, bins='log')
        fig.suptitle(label + ' Q-only DFS')
        fig.savefig(label+'_DFSq_log10PWV.png')

        fig = scatterhex_hist_DFS_PWV(
            pwv, dt, logx=True, bins='log')
        fig.suptitle(label + ' T-only DFS')
        fig.savefig(label+'_DFSt_log10PWV.png')

        fig = scatterhex_hist_DFS_PWV(
            pwv, dtq, logx=True, bins='log')
        fig.suptitle(label + ' joint TQ DFS')
        fig.savefig(label+'_DFStq_log10PWV.png')

        fig = scatterhex_hist_DFS_PWV(
            pwv, dq, logx=False, bins='log')
        fig.suptitle(label + ' Q-only DFS')        
        fig.savefig(label+'_DFSq_PWV.png')

        fig = scatterhex_hist_DFS_PWV(
            pwv, dt, logx=False, bins='log')
        fig.suptitle(label + ' T-only DFS')
        fig.savefig(label+'_DFSt_PWV.png')

        fig = scatterhex_hist_DFS_PWV(
            pwv, dtq, logx=False, bins='log')
        fig.suptitle(label + ' joint TQ DFS')
        fig.savefig(label+'_DFStq_PWV.png')


if __name__ == "__main__":
    vdat = {}
    #choose file to read. 
    #filename = '/data/users/nnn/ERA5/h5/ERA5_DFS_pwv_antarctica.h5'
    #filename = '/data/users/nnn/ERA5/h5/ERA5_DFS_pwv_ocean.h5'
    #filename = '/data/users/nnn/ERA5/h5/ERA5_DFS_pwv_Greenland.h5'

    #these files should have the diagonal values of the A matrix
    #filename = '/data/users/nnn/ERA5/h5/ERA5_DFS_diag_pwv_Ocean.h5'
    #filename = '/data/users/nnn/ERA5/h5/ERA5_DFS_diag_pwv_Greenland.h5'
    #filename = '/data/users/nnn/ERA5/h5/ERA5_DFS_diag_pwv_Antarctica.h5
    filename = '/data/users/nnn/DFS_analysis/ERA5_DFS_diag_pwv_srf9p2_Tropics.h5'
    print(filename)

    with h5py.File(filename,'r') as h:
        for dataset in h:
            vdat[dataset] = h[dataset][:]

    dt = vdat['dfs_temp']
    dq = vdat['dfs_q']
    dtq = vdat['dfs_tq']
    pwvi = vdat['pwv_interp']

    dt_flat = dt.flatten()
    dq_flat = dq.flatten()
    dtq_flat = dtq.flatten()
    pwvi_flat = pwvi.flatten()


    scatterhex_hist_DFS_PWV(pwvi_flat, dtq_flat, logx=True)

    plt.show()

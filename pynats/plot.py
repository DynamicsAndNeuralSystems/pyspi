from pynats.calculator import Calculator
import numpy as np
import pandas as pd
from scipy import stats
from pynats import utils
import warnings

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib import rcParams

from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list, set_link_color_palette
from sklearn.decomposition import PCA

def diagnostics(calc):
    """ TODO: print out all diagnostics, e.g., compute time, failures, etc.
    """
    sid = np.argsort(calc._proctimes)
    print('Processing times for all {} measures:'.format(len(sid)))
    for i in sid:
        print('[{}] {}: {} s'.format(i,calc._measure_names[i],calc._proctimes[i]))

def truth(calc,truth):
    corrs = np.zeros((calc.n_measures))

    truthflat = truth.flatten()
    for i in range(calc.n_measures):
        adj = np.transpose(calc.adjacency[i,:,:])
        np.fill_diagonal(adj,0)
        measflat = adj.flatten()
        corrs[i] = stats.pearsonr(truthflat,measflat)[0]

    sid = np.argsort(corrs)
    print('Pearson correlations for all {} measures to truth:'.format(len(sid)))
    for i in sid:
        print('[{}] {}: {}%'.format(i,calc._measure_names[i],100*corrs[i]))

def update_plots(num,data,lines,indicator):
    maxT = data.shape[1]
    for t, line in enumerate(lines):
        line.set_ydata(data[:,(num+t)%maxT])
        te = ((num+t)%maxT)+1
        indicator[t].set_ydata([te-1, te])

def plot_spacetime(data,cmap='Greens',window=7,cluster=True):
    dat = data.to_numpy(squeeze=True)

    if cluster:
        g = sns.clustermap(np.transpose(dat),
                                cmap=cmap, figsize=(10,10),
                                row_cluster=False, cbar_pos=None )
    else:
        g = sns.clustermap(np.transpose(dat),
                                cmap=cmap, figsize=(10,10),
                                col_cluster=False, row_cluster=False,
                                cbar_pos=None )

    ax_im = g.ax_heatmap
    fig = ax_im.figure
    g.gs.update(left=0.05, right=0.45, bottom=0.1, top=0.9)
    ax_im.set_xlim([0,data.n_processes+1])
    ax_im.set_facecolor((1.0, 1.0, 1.0))
    ax_im.set_xlabel('Process')
    ax_im.set_ylabel('Time')
    ax_im.set_title(f'Dataset: {data.name}')
    ax_im.set_xticks(np.linspace(0,data.n_processes,5,dtype=int).tolist(),
                        np.linspace(0,data.n_processes,5,dtype=str).tolist())
    ax_im.set_yticks(np.linspace(0,data.n_observations,5,dtype=int).tolist(),
                        np.linspace(0,data.n_observations,5,dtype=str).tolist())

    gs2 = gridspec.GridSpec(1,1, left=0.6, right=0.9, bottom=0.3, top=0.55)
    ax_st = g.fig.add_subplot(gs2[0,0])

    cols = plt.cm.Blues(np.linspace(0,1,window))
    lines = []
    indicators = []
    for t in range(window):
        lines.append(ax_st.plot(dat[:,t],color=cols[t])[0])
        indicators.append(ax_im.plot([data.n_processes+1,data.n_processes+1],[t-1,t],linewidth=6,color=cols[t], alpha=0.6)[0])

    lims = [np.min(dat),np.max(dat)]
    padding = np.ptp(lims)*0.05
    ax_st.set_ylim([lims[0]-padding,lims[1]+padding])
    line_ani = animation.FuncAnimation(fig, update_plots,data.n_observations,fargs=(dat,lines,indicators),interval=100,blit=False)

    plt.show()

# TODO: only use the top nmeasures features
def heatmaps(calc,ncols=5,nmeasures=None,split=False,cmap='PiYG'):
    if nmeasures is None:
        nmeasures = calc.n_measures
    nrows = 1 + nmeasures // ncols

    _, axs = plt.subplots(nrows,ncols,sharex=True,sharey=True,squeeze=False)

    lw = 0.0
    if split is True:
        lw = 0.01

    for i in range(nmeasures):
        ccol = i % ncols
        crow = i // ncols
        myax = axs[crow,ccol]

        # Ensures colorbar and plot are same height:
        divider = make_axes_locatable(myax)
        cax = divider.append_axes("right", size="5%", pad=0.1)    
        
        adj = calc.adjacency[i,:,:]
        sns.heatmap(adj,
                    ax=myax, cbar_ax=cax,
                    square=True, linewidth=lw,
                    cmap=cmap, mask=np.invert(np.isnan(adj)), center=0.00)
        myax.set_title('[' + str(i) + '] ' + utils.strshort(calc._measure_names[i],20))

    # Make remaining subplots empty:
    for ax in axs[-1,(nmeasures % ncols):]:
        ax.axis('off')

def clustermap(calc,which_measure='all',carpet_plot=False,sort_carpet=True,categories=None,strtrunc=None,cmap='PiYG',**kwargs):

    if carpet_plot is True:
        figsize = (10,5)
    else:
        figsize = (10,10)

    if isinstance(which_measure,int):

        adj = calc.adjacency[which_measure,:,:]
        np.fill_diagonal(adj, 0)
        adj = np.nan_to_num(adj)

        cat_colors = None
        if categories is not None:
            cats = pd.Series(categories)
            # Create a categorical palette to identify the networks
            category_pal = sns.color_palette('pastel',len(categories))
            category_lut = dict(zip(cats.unique(), category_pal))
            cat_colors = cats.map(category_lut).tolist()

        g = sns.clustermap(adj, cmap=cmap,
                            center=0.0, figsize=figsize,
                            col_colors=cat_colors, row_colors=cat_colors,**kwargs,
                            cbar_pos=(0, .2, .03, .4) )

        ax = g.ax_heatmap
    elif which_measure == 'all':
        adj = np.moveaxis(calc.adjacency,0,2)
        adjs = np.resize(adj,(calc.adjacency.shape[1]**2,calc.adjacency.shape[0]))
        if strtrunc is not None:
            colnames = utils.strshort(calc._measure_names,strtrunc)
        else:
            colnames = calc._measure_names
        df = pd.DataFrame(adjs,columns=colnames)
        
        df.fillna(0,inplace=True)
        corrs = df.corr(method='spearman')
        corrs.fillna(0,inplace=True)
        g = sns.clustermap(corrs, cmap=cmap,
                            center=0.0, figsize=figsize,
                            **kwargs, xticklabels=1, yticklabels=1,
                            cbar_pos=(0, .2, .03, .4) )
        ax = g.ax_heatmap

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    if carpet_plot is True:

        g.gs.update(left=0.55, right=0.8)

        dat = calc.dataset.to_numpy(squeeze=True)
        if sort_carpet is True:
            Z = linkage(dat,method='average')
            dat = dat[leaves_list(Z),:]
            #create new gridspec for the right part
            gs2 = gridspec.GridSpec(2,2, left=0.05, right=0.45, top=0.85, height_ratios=[0.1, 1], width_ratios=[1, 0.1], hspace=0.5)
        else:
            gs2 = gridspec.GridSpec(2,1, left=0.05, right=0.45, top=0.85, height_ratios=[0.1, 1], hspace=0.5)

        # create axes within this new gridspec
        ax2 = g.fig.add_subplot(gs2[1,0])
        # plot boxplot in the new axes
        img = ax2.imshow(dat,aspect='auto',interpolation='none',cmap='Greens')

        if sort_carpet is True:
            ax3 = g.fig.add_subplot(gs2[1,1])
            rcParams['lines.linewidth'] = 0.5
            dendrogram(Z, ax=ax3,orientation='right',
                            no_labels=True, color_threshold=0,
                            above_threshold_color='k')
            ax3.axis('off')
            ax2.set_yticks(range(0,dat.shape[0]))
            ax2.set_yticklabels([str(x) for x in leaves_list(Z)])

        cax = g.fig.add_subplot(gs2[0,0])
        g.fig.colorbar(img, cax=cax, orientation='horizontal')
        cax.xaxis.set_ticks_position('top')

        ax2.set_xlabel('Time')
        ax2.set_ylabel('Process')
        cax.set_title(f'Dataset: {calc.dataset.name}')

# TODO: only use the top nmeasures features
def flatten(calc,nmeasures=None,split=False,normalise=True,cluster=True,row_cluster=False,strtrunc=None,cmap='PiYG',plot=True):
    if nmeasures is None:
        nmeasures = calc.n_measures 

    il = np.tril_indices(calc._dataset.n_processes,-1)

    ytickl = [str(il[0][i]) + '->' + str(il[1][i]) for i in range(len(il[0]))]
    yticku = [str(il[1][i]) + '->' + str(il[0][i]) for i in range(len(il[0]))]

    yticks = [0 for i in range(2*len(il[0]))]
    yticks[slice(0,-1,2)] = yticku
    yticks[slice(1,len(yticks),2)] = ytickl

    yticks = yticks
    if strtrunc is not None:
        xticks = utils.strshort(calc._measure_names,strtrunc)
    else:
        xticks = calc._measure_names

    pwarr = np.empty((2*len(il[0]),nmeasures))
    for f in range(nmeasures):
        pwarr[slice(0,-1,2),f] = calc.adjacency[f,il[1],il[0]]
        pwarr[slice(1,len(yticks),2),f] = calc.adjacency[f,il[0],il[1]]

    if normalise is True:
        pwarr = stats.zscore(pwarr,axis=0,nan_policy='omit')

        # For the positive measures make sure we scale these properl
        for meas in [i for i in range(nmeasures) if calc._measures[i].ispositive() is True]:
            pwarr[:,meas] = ( pwarr[:,meas] - np.nanmin(pwarr[:,meas]) ) / 2

    df = pd.DataFrame(pwarr, index=yticks, columns=xticks)

    df.columns.name = 'Pairwise measure'
    df.index.name = 'Processes'
    if plot:
        if cluster is True:
            dfp = df
            dfp.fillna(0,inplace=True)
            try:
                g = sns.clustermap(dfp,
                                    cmap=cmap, row_cluster=row_cluster,
                                    figsize=(7, 7), dendrogram_ratio=(.2, .2),
                                    xticklabels=1 )
                ax = g.ax_heatmap
            except FloatingPointError:
                warnings.warn('Disimilarity value returned NaN. Have you run .prune()?',RuntimeWarning)
                return
        else:
            _, ax = plt.subplots(1,1)
            ax = sns.heatmap(df, ax=ax, cmap=cmap,
                                xticklabels=1 )
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    return df

def statespace(cf):
    pca = PCA(n_components=2, svd_solver='full')
    _, ax = plt.subplots(1,1)
    for i in cf.calculators.index:
        calc = cf.calculators.loc[i][0]
        df = flatten(calc,plot=False)
        arr = df.to_numpy(na_value=0)
        z = pca.fit(arr).singular_values_
        plt.plot(z[0],z[1], 'ro')
        plt.text(z[0]+1,z[1], calc.name, fontsize=11)
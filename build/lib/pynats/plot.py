from pynats.calculator import Calculator
from pynats.container import CalculatorFrame

import numpy as np
import pandas as pd
from scipy import stats
from pynats import utils
import warnings

import matplotlib as mpl
mpl.use('GTK3Agg') # TKAgg has problems with multithreading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib import rcParams

import pynats.lib.ScientificColourMaps6 as SCM6
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list, set_link_color_palette
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from sklearn.manifold import TSNE

def asframe(func):
    def convert(calcs,**kwargs):
        if isinstance(calcs,Calculator):
            cf = CalculatorFrame()
            cf.add_calculator(calcs)
            return func(cf,**kwargs)
        if isinstance(calcs,list) and isinstance(calcs[0],Calculator):
            cf = CalculatorFrame(calculators=calcs)
            return func(cf,**kwargs)
        elif isinstance(calcs,CalculatorFrame):
            return func(calcs,**kwargs)
        else:
            raise TypeError('First parameter must be either a list of Calculators or a CalculatorFrame.')

    return convert

def diagnostics(calc):
    """ TODO: print out all diagnostics, e.g., compute time, failures, etc.
    """
    sid = np.argsort(calc._proctimes)
    print(f'Processing times for all {len(sid)} measures:')
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
    print(f'Pearson correlations for all {len(sid)} measures to truth:')
    for i in sid:
        print('[{}] {}: {}%'.format(i,calc._measure_names[i],100*corrs[i]))

def update_plots(num,data,lines):
    maxT = data.shape[1]
    for t, line in enumerate(lines):
        line.set_ydata(data[:,(num+t)%maxT])
        te = ((num+t)%maxT)+1

def plot_spacetime(data,cmap='davos_r',window=7,cluster=True,savefilename=None):
    dat = data.to_numpy(squeeze=True)

    g = sns.clustermap(np.transpose(dat),
                            cmap=cmap, figsize=(10,10),
                            col_cluster=cluster, row_cluster=False,
                            cbar_pos=None )

    ax_im = g.ax_heatmap
    fig = ax_im.figure
    g.gs.update(left=0.05, right=0.45, bottom=0.1, top=0.9)
    ax_im.set_xlabel('Process')
    ax_im.set_ylabel('Time')
    ax_im.figure.suptitle(f'Space-time amplitude plot for "{data.name}"')

    gs2 = gridspec.GridSpec(1,1, left=0.6, right=0.9, bottom=0.3, top=0.55)
    ax_st = g.fig.add_subplot(gs2[0,0])

    cols = plt.get_cmap('devon_r',window)
    lines = []
    for t in range(window):
        lines.append(ax_st.plot(dat[:,t],color=cols(t))[0])

    lims = [np.min(dat),np.max(dat)]
    padding = np.ptp(lims)*0.05
    ax_st.set_ylim([lims[0]-padding,lims[1]+padding])
    line_ani = animation.FuncAnimation(fig,update_plots,data.n_observations,fargs=(dat,lines),interval=100,blit=False)

    ax_im.locator_params(axis='y', nbins=6)
    if savefilename is not None:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        line_ani.save(savefilename, writer=writer)
    else:
        plt.show()

# TODO: only use the top nmeasures features
def heatmaps(calc,ncols=5,nmeasures=None,cmap=sns.color_palette("coolwarm", as_cmap=True),**kwargs):
    if nmeasures is None:
        nmeasures = calc.n_measures
    nrows = 1 + nmeasures // ncols

    _, axs = plt.subplots(nrows,ncols,sharex=True,sharey=True,squeeze=False)

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
                    square=True,
                    cmap=cmap, mask=np.invert(np.isnan(adj)), center=0.00, **kwargs)
        myax.set_title('[' + str(i) + '] ' + utils.strshort(calc._measure_names[i],20))

    # Make remaining subplots empty:
    for ax in axs[-1,(nmeasures % ncols):]:
        ax.axis('off')

def clustermap(calc,which_measure=None,plot=True,plot_data=False,sort_data=True,categories=None,strtrunc=20,data_cmap='devon_r',clustermap_kwargs={'cmap': sns.color_palette("coolwarm", as_cmap=True)}):

    if plot_data is True:
        figsize = (15,10)
    else:
        figsize = (10,10)

    if isinstance(which_measure,int) and which_measure > 0:
        corrs = calc.adjacency[which_measure,:,:]
        np.fill_diagonal(corrs, 0)
        corrs = np.nan_to_num(corrs)

        if plot is True:
            cat_colors = None
            if categories is not None:
                cats = pd.Series(categories)
                # Create a categorical palette to identify the networks
                category_pal = sns.color_palette('pastel',len(categories))
                category_lut = dict(zip(cats.unique(), category_pal))
                cat_colors = cats.map(category_lut).tolist()

            g = sns.clustermap(corrs, center=0.0, figsize=figsize,
                                col_colors=cat_colors, row_colors=cat_colors,**clustermap_kwargs,
                                dendrogram_ratio=.05 )
            ax_hm = g.ax_heatmap
            ax_hmcb = g.ax_cbar
    else:
        adj = np.moveaxis(calc.adjacency,0,2)
        adjs = np.resize(adj,(calc.adjacency.shape[1]**2,calc.adjacency.shape[0]))
        if strtrunc is not None:
            colnames = utils.strshort(calc._measure_names,strtrunc)
        else:
            colnames = calc._measure_names
        df = pd.DataFrame(adjs,columns=colnames)
        corrs = df.corr(method='spearman')

        if plot is True:
            mask = np.isnan(corrs.to_numpy())
            corrs = corrs.fillna(0)

            if corrs.shape[0] > 20:
                sns.set(font_scale=0.7)
            g = sns.clustermap(corrs, mask=mask,
                                center=0.0, figsize=figsize,
                                **clustermap_kwargs, xticklabels=1, yticklabels=1 )
            ax_hm = g.ax_heatmap
            ax_hmcb = g.ax_cbar

    sns.set(font_scale=1.0)

    if plot is True:
        plt.setp(ax_hm.xaxis.get_majorticklabels(), rotation=45, ha='right')

        if plot_data is True:

            g.gs.update(left=0.33, right=0.8, top=0.9)

            dat = calc.dataset.to_numpy(squeeze=True)
            if sort_data is True:
                Z = linkage(dat,method='average')
                dat = dat[leaves_list(Z),:]
                #create new gridspec for the right part
                gs2 = gridspec.GridSpec(44,1, left=0.1, right=0.25, top=0.85, bottom=0.2, hspace=0.0)
            else:
                gs2 = gridspec.GridSpec(40,1, left=0.1, right=0.25, top=0.85, bottom=0.2, hspace=0.0)

            ax_saplot = g.fig.add_subplot(gs2[6:40,0])
            img = ax_saplot.imshow(np.transpose(dat),aspect='auto',interpolation='none',cmap=data_cmap)

            ax_saplot.xaxis.set_label_position('top') 
            ax_saplot.xaxis.set_ticks_position('top') 
            ax_saplot.set_xlabel('Process')
            ax_saplot.set_ylabel('Time')

            # create axes within this new gridspec
            if sort_data is True:
                ax_dend = g.fig.add_subplot(gs2[40:,0])
                rcParams['lines.linewidth'] = 0.5
                dendrogram(Z, ax=ax_dend, orientation='bottom', no_labels=True, color_threshold=0,
                                above_threshold_color='k')
                ax_dend.axis('off')

                ax_saplot.set_xticks(range(0,dat.shape[0]))
                ax_saplot.set_xticklabels([str(x) for x in leaves_list(Z)])

            ax_saplot.grid(False)

            cb_ax = g.fig.add_subplot(gs2[1,0])
            g.fig.colorbar(img, cax=cb_ax, orientation='horizontal')
            cb_ax.xaxis.set_ticks_position('top')
            plt.subplots_adjust(bottom=0.2)
            ax_hmcb.set_position([0.35, 0.8, 0.02, 0.1])

        g.fig.suptitle(f'Clustermap for "{calc.dataset.name}"')
        return corrs, g.fig
    return corrs

def flatten(calc,nmeasures=None,split=False,transformer=None,cluster=True,row_cluster=False,strtrunc=None,cmap=sns.color_palette("coolwarm", as_cmap=True),plot=True,**kwargs):
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

    if transformer is not None:
        pwarr = transformer.fit_transform(pwarr)

    df = pd.DataFrame(pwarr, index=yticks, columns=xticks)

    df.columns.name = 'Pairwise measure'
    df.index.name = 'Processes'
    if plot:
        if cluster is True:
            dfp = df
            dfp.fillna(0,inplace=True)
            try:
                sns.set(font_scale=0.7)
                g = sns.clustermap(dfp,
                                    cmap=cmap, row_cluster=row_cluster,
                                    figsize=(15, 7),
                                    xticklabels=1, **kwargs )
                sns.set(font_scale=1.0)
                ax = g.ax_heatmap
            except FloatingPointError:
                warnings.warn('Disimilarity value returned NaN. Have you run .prune()?',RuntimeWarning)
                return
            ax_hmcb = g.ax_cbar
            plt.subplots_adjust(left=0.0,right=0.9,bottom=0.2,top=0.9)
            ax_hmcb.set_position([0.1, 0.25, 0.015, 0.4])
            ax_hmcb.yaxis.set_ticks_position('left') 
        else:
            _, ax = plt.subplots(1,1)
            ax = sns.heatmap(df, ax=ax, cmap=cmap,
                                xticklabels=1, **kwargs )
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        return df, ax.figure    
    return df
       
@asframe
def flattenall(cf,plot=True,cluster=True,cmap=sns.color_palette("coolwarm", as_cmap=True),row_cluster=False,flatten_kwargs={},**kwargs):
    df = pd.DataFrame()
    for i in cf.calculators.index:
        df2 = flatten(cf.calculators.loc[i][0],strtrunc=None,plot=False,**flatten_kwargs)
        df = pd.concat([df, df2], axis=0, sort=False, ignore_index=True)

    if plot:
        if cluster is True:
            dfp = df
            dfp.fillna(0,inplace=True)
            try:
                sns.set(font_scale=0.7)
                g = sns.clustermap(dfp,
                                    cmap=cmap, row_cluster=row_cluster,
                                    figsize=(15, 7),
                                    xticklabels=1, **kwargs )
                sns.set(font_scale=1.0)
                ax = g.ax_heatmap
            except FloatingPointError:
                warnings.warn('Disimilarity value returned NaN. Have you run prune() for the calculator?', RuntimeWarning)
                return
            ax_hmcb = g.ax_cbar
            plt.subplots_adjust(left=0.0,right=0.9,bottom=0.2,top=0.9)
            ax_hmcb.set_position([0.1, 0.25, 0.015, 0.4])
            ax_hmcb.yaxis.set_ticks_position('left') 
        else:
            _, ax = plt.subplots(1,1)
            ax = sns.heatmap(df, ax=ax, cmap=cmap,
                                xticklabels=1, **kwargs )
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        return df, ax.figure    

    return df

@asframe
def clusterall(cf,approach='mean',plot=True,reducer=TSNE(n_components=1),flatten_kwargs={},clustermap_kwargs={'cmap': sns.color_palette("coolwarm", as_cmap=True)}):
    if approach == 'flatten':
        df = flattenall(cf,plot=False,flatten_kwargs=flatten_kwargs)
        df.fillna(0,inplace=True)
        corrs = df.corr(method='spearman')
        corrs.fillna(0,inplace=True)
    else:
        df = pd.DataFrame()
        for i in cf.calculators.index:
            df2 = flatten(cf.calculators.loc[i][0],plot=False,strtrunc=None,**flatten_kwargs).corr(method='spearman')
            if df.shape[0] > 0:
                df = pd.concat([df, df2], axis=0, sort=False)
            else:
                df = df2
        
        corrs = df.groupby('Pairwise measure').mean().reindex(df.keys())
        corrs.fillna(0,inplace=True)

    if plot:
        if corrs.shape[0] > 20:
            sns.set(font_scale=0.7)
        g = sns.clustermap(corrs, center=0.0, xticklabels=1, yticklabels=1,**clustermap_kwargs)
        ax = g.ax_heatmap
        ax_hmcb = g.ax_cbar
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        sns.set(font_scale=1)
        g.gs.update(top=0.9)
        g.fig.suptitle(f'Clustermap for all {len(cf.calculators.index)} datasets of "{cf.name}" frame')
        ax_hmcb.set_position([0.05, 0.8, 0.02, 0.1])
        
        return corrs, ax.figure
    return corrs

@asframe
def dataspace(cf):
    pca = PCA(n_components=2, svd_solver='full')
    _, ax = plt.subplots(1,1)
    dfs = {}
    for i in cf.calculators.index:
        calc = cf.calculators.loc[i][0]
        dfs[calc.name] = flatten(calc,plot=False)
        arr = dfs[calc.name].to_numpy(na_value=0)
        z = pca.fit(arr).singular_values_
        plt.plot(z[0],z[1], 'ro')
        plt.text(z[0]+1,z[1], calc.name, fontsize=11)

    return dfs, ax.figure

@asframe
def measurespace(cf,averaged=False,pairplot=False,jointplot=False,clustermap=False,reducer=UMAP(),flatten_kwargs={},clustermap_kwargs={}):

    # Reducer is any dimensionality reduction algorithm that calls fit_transform, e.g.,
    # reducer = MDS(n_components=2,dissimilarity='precomputed')
    # or reducer = TSNE(n_components=2)

    if averaged:
        df = clusterall(cf,approach='flatten',plot=False)
        n_measures = df.shape[0]

        embedding = reducer.fit_transform(1-df.to_numpy())

        if pairplot:
            sns.pairplot(flattenall(cf,plot=False))
    else:
        dfs = {}
        for i, _index in enumerate(cf.calculators.index):
            calc = cf.calculators.loc[_index][0]
            dfs[calc.name] = flatten(calc,plot=False,**flatten_kwargs)
        df = pd.concat(dfs,axis=1,join='inner')
        embedding = reducer.fit_transform(np.transpose(df.to_numpy()))
        n_measures = len(set([m[1] for m in df.columns]))
        
    cmap = sns.color_palette("husl", n_measures)

    bad_markers = ['.', ',', '8', 'H']
    marker_list = list(plt.matplotlib.lines.Line2D.markers.keys())
    marker_list = [item for item in marker_list if item not in bad_markers] # Remove the ones that are hard to see

    if not averaged and jointplot or clustermap:
        jointdf = pd.DataFrame()
        for i in range(embedding.shape[0]):
            dataset = df.columns[i][0]
            measure = df.columns[i][1]
            s = pd.Series([measure,embedding[i,0],embedding[i,1]],['measure','e1','e2'])
            jointdf = jointdf.append(s,ignore_index=True)

        if jointplot:
            g = sns.jointplot(data=jointdf, x='e1', y='e2', hue='measure', palette=cmap)
        else:
            embeddf = pd.DataFrame(data=embedding[:,1],index=df.columns).unstack().transpose()
            embeddf.dropna(inplace=True)
            if embeddf.shape[0] > 20:
                sns.set(font_scale=0.7)
            g = sns.clustermap(embeddf, center=0.0, xticklabels=1, yticklabels=1,**clustermap_kwargs)
            ax2 = g.ax_heatmap
            ax_hmcb = g.ax_cbar
            # plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            sns.set(font_scale=1)
            g.gs.update(top=0.9)
            g.fig.suptitle(f'Clustermap for all {len(cf.calculators.index)} datasets')
            ax_hmcb.set_position([0.05, 0.8, 0.02, 0.1])

        fig = g.fig
    else:
        if averaged:
            fig = plt.figure(figsize=(6.75,5))
        else:
            fig = plt.figure(figsize=(8,5))

        ax = fig.add_subplot()

        colours = {}
        markers = {}
        leg_phs = []
        leg_labels = []

        leg2_phs = []
        leg2_labels = []
        for i in range(embedding.shape[0]):
            if averaged:
                dataset = 'all'
                measure = df.columns[i]
            else:
                dataset = df.columns[i][0]
                measure = df.columns[i][1]

            try:
                marker = markers[dataset]
            except KeyError as err:
                marker = marker_list[len(markers)]
                markers[dataset] = marker
                ph, = ax.plot([],[],marker=marker, color=(0,0,0,0.75), linestyle="None", markersize=6)
                leg2_phs.append(ph)
                leg2_labels.append(utils.strshort(dataset,15))

            try:
                color = colours[measure]
            except KeyError as err:
                color = cmap[len(colours)]
                colours[measure] = color[:3] + (.75,)
                ph, = ax.plot([],[],marker='.', color=color,linestyle="None",markersize=11)
                leg_phs.append(ph)
                leg_labels.append(utils.strshort(measure,15))

            ax.plot(embedding[i,0],embedding[i,1],marker=marker,color=color,markersize=10)

        reducer_str = str(reducer).split('(')[0]
        ax.set_xlabel(f'{reducer_str} dim-1')
        ax.set_ylabel(f'{reducer_str} dim-2')

        leg = ax.legend(leg_phs,leg_labels,bbox_to_anchor=(1.375,1),loc='upper right',fontsize='xx-small')
        if not averaged:
            ax.legend(leg2_phs,leg2_labels,bbox_to_anchor=(1.3755,1),loc='upper left',fontsize='xx-small')
            ax.add_artist(leg)

        ax.grid(True)
        ax.set_title(f'Low-dimensional measure space via {str(reducer_str)}')

    fig.tight_layout()
    return df, fig

def relate(cf,meas0,meas1,raw=False,flatten_kwargs={}):

    df = pd.DataFrame()
    corr_str = f'corr({meas0}, {meas1})'
    for i, _index in enumerate(cf.calculators.index):
        calc = cf.calculators.loc[_index][0]
        try:
            rho = flatten(calc,plot=False,strtrunc=None,**flatten_kwargs)[[meas0,meas1]].corr(method='spearman').to_numpy()[0,1]
            df = df.append({corr_str: rho},ignore_index=True)
        except KeyError:
            print(f'Received key error for calculator "{calc.name}": {KeyError}')
    
    splot = sns.displot(df,x=corr_str,kde=True)
    plt.xlim(-1, 1)
    
    return df, splot.fig
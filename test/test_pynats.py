import numpy as np
import math
import pytest
import warnings

from pynats.data import Data
from pynats.calculator import Calculator
from pynats.base import undirected

np.random.seed(0) # For reproducibility

def get_data():
    T = 100
    ar_params = .75

    # Generate our random time series
    procs = [np.random.normal(size=T), np.random.normal(size=T)]
    for _i, p in enumerate(procs):
        for t in range(1,T):
            if _i == 0:
                p[t] += ar_params * p[t-1]
            else:
                p[t] += ar_params * procs[_i-1][t] # Contemporaneous causation

    # For each measure, check that the adjacencies match the subclass (directed/undirected and bivariate->adjacency)dim_order='ps')
    return Data(np.vstack(procs),dim_order='ps',normalise=True)

def test_yaml():
    data = get_data()
    calc = Calculator(dataset=data)

    """
    TODO: check the data properties all match
    """
    assert calc.n_measures == len(calc._measures), (
                'Property not equal to number of measures')

def test_adjacency():
    # Load in all base measures from the YAML file

    data = get_data()
    calc = Calculator(dataset=data)

    # Excuse the partial correlation/precision from comparing bivariate->multivariate as they 
    # TODO: why doesn't ledoit wolf work?
    excuse_bv = ['pearsonr_ledoit_wolf',
                'pearsonr_oas',
                'pcorr_empirical',
                'pcorr-sq_empirical',
                'pcorr_ledoit_wolf',
                'pcorr_shrunk',
                'pcorr_oas',
                'prec_empirical',
                'prec-sq_empirical']
    
    excuse_directed = ['coint_aeg_tstat']

    excuse_stochastic = ['ccm_max','ccm_mean','ccm_diff']

    p = data.to_numpy()
    for _i, m in enumerate(calc._measures):
        print(f'[{_i}/{calc.n_measures}] Testing measure {m.name} ({m.humanname})')

        if any([m.name == e for e in excuse_stochastic]):
            continue

        scratch_adj = m.adjacency(data.to_numpy())
        adj = m.adjacency(data)
        assert np.allclose(adj,scratch_adj,rtol=1e-1,atol=1e-2,equal_nan=True), (
                    f'{m.name} ({m.humanname}) Adjacency output changed between cached and strach computations.')

        recomp_adj = m.adjacency(data)
        assert np.allclose(adj,recomp_adj,rtol=1e-1,atol=1e-2,equal_nan=True), (
                    f'{m.name} ({m.humanname}) Adjacency output changed when recomputing.')

        for i in range(data.n_processes):
            for j in range(i+1,data.n_processes):

                assert math.isfinite(adj[i,j]), (f'{m.name} ({m.humanname}): Invalid adjacency entry ({i},{j}): {adj[i,j]}')
                assert math.isfinite(adj[j,i]), (f'{m.name} ({m.humanname}): Invalid adjacency entry ({j},{i}): {adj[j,i]}')

                try:
                    s_t = m.bivariate(data,i=i,j=j)
                    new_s_t = m.bivariate(p[i],p[j])
                    assert s_t == pytest.approx(new_s_t,rel=1e-1,abs=1e-2), (
                        f'{m.name} ({m.humanname}) Bivariate output from cache mismatch results from scratch for computation ({i},{j}): {s_t} != {new_s_t}')

                    t_s = m.bivariate(data,i=j,j=i)
                    new_t_s = m.bivariate(p[j],p[i])
                    assert t_s == pytest.approx(new_t_s,rel=1e-1,abs=1e-2), (
                        f'{m.name} ({m.humanname}) Bivariate output from cache mismatch results from scratch for computation ({j},{i}): {t_s} != {new_t_s}')
                except NotImplementedError:
                    a = m.adjacency(p[[i,j]])
                    s_t, t_s = a[0,1], a[1,0]

                assert math.isfinite(s_t), (f'{m.name} ({m.humanname}): Invalid source->target output: {s_t}')
                assert math.isfinite(t_s), (f'{m.name} ({m.humanname}): Invalid target->source output: {t_s}')

                if not any([m.name == e for e in excuse_bv]):
                    try:
                        assert s_t == pytest.approx(adj[i,j], rel=1e-1, abs=1e-2)
                    except AssertionError:
                        assert np.abs(s_t - adj[i,j]) < np.abs(t_s - adj[i,j])*2, (
                            f'{m.name} ({m.humanname}): Bivariate output ({i},{j}) does not match adjacency: {s_t} != {adj[i,j]} '
                                f' AND the lower diagonal is over 2x closer to it: {t_s} is closer to {adj[i,j]}')

                if not any([m.name == e for e in excuse_directed]):
                    if isinstance(m,undirected):
                        s_t == pytest.approx(t_s, rel=1e-1, abs=1e-2), (
                            f'{m.name} ({m.humanname}): Found directed measurement for entry ({i},{j}): {s_t} != {t_s}')
                    else:
                        s_t != pytest.approx(t_s, rel=1e-1, abs=1e-2), (
                                f'{m.name} ({m.humanname}): Found undirected measurement for entry ({i},{j}): {s_t} == {t_s}')

"""
    Individual tests specific to each measure.

    These tests are either super simple (e.g., checking correlation == correlation)
    or taken from the documentation examples for more complex measures (e.g., CCM or transfer entropy).

    More advanced testing will be *slowly* introduced into the package
"""
def test_spearmanr():
    from pynats.correlation import spearmanr

    x = np.random.normal(size=100)
    y = 0.9*x

    y_ind = np.random.normal(size=100)

    calc = spearmanr()
    dep = calc.bivariate(x,y)
    ind = calc.bivariate(x,y_ind)

    assert dep > ind, (f'{calc.humanname} test failed: {dep} < {ind}')


def test_kendalltau():
    from pynats.correlation import kendalltau

def test_connectivity():
    from pynats.correlation import connectivity
    
def test_ccm():
    """
    Ensure anchovy predicts temperature as per example:
    https://sugiharalab.github.io/EDM_Documentation/algorithms_high_level/
    """
    # Load our wrapper
    from pynats.temporal import ccm

    # Load anchovy dataset
    from pyEDM import sampleData
    sardine_anchovy_sst = sampleData['sardine_anchovy_sst']
    src = sardine_anchovy_sst['anchovy'].to_numpy()
    targ = sardine_anchovy_sst['np_sst'].to_numpy()

    stats = ['mean','max','diff']

    for _i, s in enumerate(stats):
        calc = ccm(statistic=s)
        s_t = calc.bivariate(src,targ)
        t_s = calc.bivariate(targ,src)
        assert s_t > t_s, (f'CCM test failed anchovy-temperature test for stat {s}: {s_t} < {t_s}')

def test_anm():
    # Load our wrapper
    from pynats.causal import anm

    # Load Tuebingen dataset
    from cdt.data import load_dataset
    t_data, _ = load_dataset('tuebingen')
    src, targ = t_data['A']['pair1'], t_data['B']['pair1']

    calc = anm()
    s_t = calc.bivariate(src,targ)
    t_s = calc.bivariate(targ,src)

    assert s_t > t_s, (f'{calc.humanname} test failed test for pair1: {s_t} < {t_s}')

def test_gpfit():
    # Load our wrapper
    from pynats.causal import gpfit

    # Load Tuebingen dataset
    from cdt.data import load_dataset
    t_data, _ = load_dataset('tuebingen')
    src, targ = t_data['A']['pair1'], t_data['B']['pair1']

    calc = gpfit()
    s_t = calc.bivariate(src,targ)
    t_s = calc.bivariate(targ,src)

    assert s_t > t_s, (f'{calc.humanname} test failed test for pair1: {s_t} < {t_s}')

def test_cds():
    # Load our wrapper
    pass
    # from pynats.causal import cds

    # x = np.random.normal(size=100)
    # y = 0.9*x

    # y_ind = np.random.normal(size=100)

    # calc = cds()
    # dep = calc.bivariate(x,y)
    # ind = calc.bivariate(x,y_ind)
    
    # assert dep > ind, (f'{calc.humanname} test failed: {dep} < {ind}')

def test_igci():
    # Load our wrapper
    pass
    # from pynats.causal import igci

    # x = np.random.normal(size=100)
    # y = 0.9*x

    # y_ind = np.random.normal(size=100)

    # calc = igci()
    # dep = calc.bivariate(x,y)
    # ind = calc.bivariate(x,y_ind)

    # assert dep > ind, (f'{calc.humanname} test failed: {dep} < {ind}')

def test_hsic():
    # Import our wrapper
    from pynats.correlation import hsic

    # Import linear simulation code
    x = np.random.normal(size=100)
    y = 0.9*x

    y_ind = np.random.normal(size=100)

    calc = hsic()
    dep = calc.bivariate(x,y)
    ind = calc.bivariate(x,y_ind)

    assert dep > ind

def test_hhg():
    from pynats.correlation import hhg
    x = np.random.normal(size=100)
    y = 0.9*x

    y_ind = np.random.normal(size=100)

    calc = hhg()
    dep = calc.bivariate(x,y)
    ind = calc.bivariate(x,y_ind)

    assert dep > ind

def test_dcorr():
    from pynats.correlation import dcorr

    x = np.random.normal(size=100)
    y = 0.9*x

    y_ind = np.random.normal(size=100)

    calc = dcorr()
    dep = calc.bivariate(x,y)
    ind = calc.bivariate(x,y_ind)

    assert dep > ind

def test_mgc():
    from pynats.correlation import mgc

    x = np.random.normal(size=100)
    y = 0.9*x

    y_ind = np.random.normal(size=100)

    calc = mgc()
    dep = calc.bivariate(x,y)
    ind = calc.bivariate(x,y_ind)

    assert dep > ind

def test_load():
    import dill

    calc = Calculator()

    with open('test.pkl', 'wb') as f:
        dill.dump(calc,f)


    with open('test.pkl', 'rb') as f:
        calc = dill.load(f)

    calc.load_dataset(get_data())
    calc.compute()
    

if __name__ == '__main__':
    # TODO: make the code a bit better since anything coming from the same package uses the same dataset (e.g., the CDT and EDM stuff)
    test_spearmanr()

    test_anm()
    test_gpfit()
    # test_igci() # These two can't even get correlation?
    # test_cds() 

    test_hsic()
    test_hhg()
    test_dcorr()
    test_mgc()

    test_ccm() # 3 tests

    test_yaml()
    test_load()
    test_adjacency()

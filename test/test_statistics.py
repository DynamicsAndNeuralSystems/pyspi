import numpy as np
import math
import pytest
import warnings

from pyspi.data import Data
from pyspi.calculator import Calculator
from pyspi.base import undirected

np.random.seed(0) # For reproducibility

def get_inddata():
    T = 100

    # Generate and return our random data
    procs = np.random.normal(size=(2,T))
    return Data(np.vstack(procs),dim_order='ps',normalise=True)

def get_data():
    T = 100

    # Generate our random time series
    procs = np.random.normal(size=(2,T))
    procs[1] += 0.5 * procs[0]

    # For each statistic, check that the adjacencies match the subclass (directed/undirected and bivariate->adjacency)dim_order='ps')
    return Data(np.vstack(procs),dim_order='ps',normalise=True)

def get_more_data():
    T = 100
    M = 3
    ar_params = .75

    # Generate our random time series
    procs = np.random.normal(size=(M,T))
    for _i, p in enumerate(procs):
        for t in range(1,T):
            if _i == 0:
                p[t] += ar_params * p[t-1]
            else:
                p[t] += ar_params * procs[_i-1][t-1] # Time-lagged correlation

    # For each statistic, check that the adjacencies match the subclass (directed/undirected and bivariate->adjacency)dim_order='ps')
    return Data(np.vstack(procs),dim_order='ps',normalise=True)

def test_yaml():
    data = get_data()
    calc = Calculator(dataset=data)

    """
    TODO: check the data properties all match
    """
    assert calc.n_statistics == len(calc._statistics), (
                'Property not equal to number of statistics')

def test_adjacency():
    # Load in all base statistics from the YAML file

    data = get_data()
    calc = Calculator(dataset=data)

    # Excuse some statistics cause they're meant for mv data
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

    excuse_stochastic = ['ccm_max','ccm_mean','ccm_diff','gd_fs-1_fmin-0-05_fmax-1-57']

    p = data.to_numpy()
    for _i, m in enumerate(calc._statistics):
        print(f'[{_i}/{calc.n_statistics}] Testing statistic {m.name} ({m.humanname})')

        if any([m.name == e for e in excuse_stochastic]):
            continue

        m.adjacency(get_more_data())

        scratch_adj = m.adjacency(data.to_numpy())
        adj = m.adjacency(data)
        assert np.allclose(adj,scratch_adj,rtol=1e-1,atol=1e-2,equal_nan=True), (
                    f'{m.name} ({m.humanname}) Adjacency output changed between cached and strach computations: {adj} != {scratch_adj}')

        recomp_adj = m.adjacency(data)
        assert np.allclose(adj,recomp_adj,rtol=1e-1,atol=1e-2,equal_nan=True), (
                    f'{m.name} ({m.humanname}) Adjacency output changed when recomputing.')

        for i in range(data.n_processes):
            for j in range(i+1,data.n_processes):

                if not math.isfinite(adj[i,j]):
                    warnings.warn(f'{m.name} ({m.humanname}): Invalid adjacency entry ({i},{j}): {adj[i,j]}')
                if not math.isfinite(adj[j,i]):
                    warnings.warn(f'{m.name} ({m.humanname}): Invalid adjacency entry ({i},{j}): {adj[j,i]}')

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

                if not math.isfinite(s_t):
                    warnings.warn(f'{m.name} ({m.humanname}): Invalid source->target output: {s_t}')
                if not math.isfinite(t_s):
                    warnings.warn(f'{m.name} ({m.humanname}): Invalid target->source output: {t_s}')

                if np.all(np.isfinite([s_t,t_s])):
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
                                f'{m.name} ({m.humanname}): Found directed statisticment for entry ({i},{j}): {s_t} != {t_s}')
                        else:
                            s_t != pytest.approx(t_s, rel=1e-1, abs=1e-2), (
                                    f'{m.name} ({m.humanname}): Found undirected statisticment for entry ({i},{j}): {s_t} == {t_s}')

"""
    Individual tests specific to each statistic.

    These tests are either super simple (e.g., checking correlation == correlation)
    or taken from the documentation examples for more complex statistics (e.g., CCM or transfer entropy).

    More advanced testing will be *slowly* introduced into the package
"""
    
def test_ccm():
    """
    Ensure anchovy predicts temperature as per example:
    https://sugiharalab.github.io/EDM_Documentation/algorithms_high_level/
    """
    # Load our wrapper
    from pyspi.statistics.causal import ccm

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


def test_load():
    import dill, os

    calc = Calculator()

    with open('test.pkl', 'wb') as f:
        dill.dump(calc,f)


    with open('test.pkl', 'rb') as f:
        calc = dill.load(f)

    calc.load_dataset(get_data())
    calc.compute()

    with open('test.pkl', 'wb') as f:
        dill.dump(calc,f)

def test_group():

    calc = Calculator(labels=['hello','world','friend'])
    calc.set_group([['hello'],['goodbye']])
    assert calc.group == 0
    assert calc.group_name == 'hello'
    
    calc.set_group([['hello','world'],['goodbye','world']])
    assert calc.group == 0
    assert calc.group_name == 'hello, world'

    calc.set_group([['hello','cruel','world'],['goodbye','world']])
    assert calc.group is None
    assert calc.group_name is None

if __name__ == '__main__':

    test_yaml()
    test_load()
    test_group()
    test_adjacency()

    # This was a bit tricky to implement so just ensuring it passes a test from the creator's website
    test_ccm()
"""Test data class."""
import numpy as np
from pyspi.data import Data
from scipy.stats import zscore

def test_data_properties():
    """Test data properties attributes."""
    n = 10
    d = Data(np.arange(n), 's', normalise=False)
    real_time = d.n_realisations_observations()
    assert (real_time == n), 'Realisations in time are not returned correctly.'
    cv = (0, 8)
    real_time = d.n_realisations_observations(current_value=cv)
    assert (real_time == (n - cv[1])), ('Realisations in time are not '
                                        'returned correctly when current value'
                                        ' is set.')


def test_set_data():
    """Test if data is written correctly into a Data instance."""
    source = np.expand_dims(np.repeat(1, 30), axis=1)
    target = np.expand_dims(np.arange(30), axis=1)

    data = Data(normalise=False)
    data.set_data(np.vstack((source.T, target.T)), 'ps')

    assert (data.to_numpy()[0, :].T == source.T).all(), ('Class data does not match '
                                                   'input (source).')
    assert (data.to_numpy()[1, :].T == target.T).all(), ('Class data does not match '
                                                   'input (target).')

    d = Data()
    data = np.arange(10000).reshape((2, 1000, 5))  # random data with correct
    d = Data(data, dim_order='psr')               # order od dimensions
    assert (d.to_numpy().shape[0] == 2), ('Class data does not match input, number '
                                    'of processes wrong.')
    assert (d.to_numpy().shape[1] == 1000), ('Class data does not match input, '
                                       'number of observations wrong.')
    assert (d.to_numpy().shape[2] == 5), ('Class data does not match input, number '
                                    'of replications wrong.')
    data = np.arange(3000).reshape((3, 1000))  # random data with incorrect
    d = Data(data, dim_order='ps')            # order of dimensions
    assert (d.to_numpy().shape[0] == 3), ('Class data does not match input, number '
                                    'of processes wrong.')
    assert (d.to_numpy().shape[1] == 1000), ('Class data does not match input, '
                                       'number of observations wrong.')
    assert (d.to_numpy().shape[2] == 1), ('Class data does not match input, number '
                                    'of replications wrong.')
    data = np.arange(5000)
    d.set_data(data, 's')
    assert (d.to_numpy().shape[0] == 1), ('Class data does not match input, number '
                                    'of processes wrong.')
    assert (d.to_numpy().shape[1] == 5000), ('Class data does not match input, '
                                       'number of observations wrong.')
    assert (d.to_numpy().shape[2] == 1), ('Class data does not match input, number '
                                    'of replications wrong.')


def test_data_normalisation():
    """Test if data are normalised correctly when stored in a Data instance."""
    a_1 = 100
    a_2 = 1000
    source = np.random.randint(a_1, size=1000)
    target = np.random.randint(a_2, size=1000)

    data = Data(normalise=True)
    data.set_data(np.vstack((source.T, target.T)), 'ps')

    source_std = zscore(source)
    target_std = zscore(target)
    assert (source_std == data.to_numpy()[0, :, 0]).all(), ('Standardising the '
                                                      'source did not work.')
    assert (target_std == data.to_numpy()[1, :, 0]).all(), ('Standardising the '
                                                      'target did not work.')


def test_data_type():
    """Test if data class always returns the correct data type."""
    # Change data type for the same object instance.
    d_int = np.random.randint(0, 10, size=(3, 50))
    orig_type = type(d_int[0][0])
    data = Data(d_int, dim_order='ps', normalise=False)
    # The concrete type depends on the platform:
    # https://mail.scipy.org/pipermail/numpy-discussion/2011-November/059261.html
    # Hence, compare against the type automatically assigned by Python or
    # against np.integer
    assert data.data_type is orig_type, 'Data type did not change.'
    assert issubclass(type(data.to_numpy()[0, 0, 0]), np.integer), (
        'Data type is not an int.')
    d_float = np.random.randn(3, 50)
    data.set_data(d_float, dim_order='ps')
    assert data.data_type is np.float64, 'Data type did not change.'
    assert issubclass(type(data.to_numpy()[0, 0, 0]), float), (
        'Data type is not a float.')

    """
    TODO: Sort out the realisations/replications dramah (currently unused feature)
    """
    # # Check if data returned by the object have the correct type.
    # d_int = np.random.randint(0, 10, size=(3, 50, 5))
    # data = Data(d_int, dim_order='psr', normalise=False)
    # real = data.get_realisations((0, 5), [(1, 1), (1, 3)])[0]
    # assert issubclass(type(real[0, 0]), np.integer), (
    #     'Realisations type is not an int.')
    # sl = data._get_data_slice(0)[0]
    # assert issubclass(type(sl[0, 0]), np.integer), (
    #     'Data slice type is not an int.')
    # settings = {'perm_type': 'random'}
    # sl_perm = data.slice_permute_observations(0, settings)[0]
    # assert issubclass(type(sl_perm[0, 0]), np.integer), (
    #     'Permuted data slice type is not an int.')
    # observations = data.permute_observations((0, 5), [(1, 1), (1, 3)], settings)[0]
    # assert issubclass(type(observations[0, 0]), np.integer), (
    #     'Permuted observations type is not an int.')


if __name__ == '__main__':
    test_data_type()
    test_data_normalisation()
    test_set_data()
    test_data_properties()

import numpy as np
import spectral_connectivity as sc

for i in range(100):
    dat = np.random.rand(200,10)
    m = sc.Multitaper(dat, sampling_frequency=1, time_halfbandwidth_product=3, start_time=0)
    c = sc.Connectivity(fourier_coefficients=m.fft(), frequencies=m.frequencies)
    print(f'Test {i}')
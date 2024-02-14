from setuptools import setup, find_packages

# http://www.diveintopython3.net/packaging.html
# https://pypi.python.org/pypi?:action=list_classifiers

with open('README.md', 'r', encoding='utf-8') as file:
    long_description = file.read()


install_requires = [
        'scikit-learn==1.0.1',
        'scipy==1.7.3',
        'numpy>=1.21.1',
        'pandas==1.5.0',
        'statsmodels==0.12.1',
        'pyyaml==5.4',
        'tqdm==4.50.2',
        'nitime==0.9',
        'hyppo==0.2.1',
        'pyEDM==1.9.3',
        'jpype1==1.2.0',
        'sktime==0.8.0',
        'dill==0.3.2',
        'spectral-connectivity==0.2.4.dev0',
        'torch==1.13.1',
        'cdt==0.5.23',
        'oct2py==5.2.0',
        'tslearn==0.5.2',
        'mne==0.23.0',
        'seaborn==0.11.0'
]

testing_extras = [
    'pytest==5.4.2',  # unittest.TestCase funkyness, see commit 77c1505ab
]


setup(
    name='pyspi-lib',
    packages=find_packages(),
    package_data={'': ['config.yaml',
                       'sonnet_config.yaml',
                        'fast_config.yaml',
                        'fabfour_config.yaml',
                        'octaveless_config.yaml',
                        'lib/jidt/infodynamics.jar',
                        'lib/PhiToolbox/Phi/phi_comp.m',
                        'lib/PhiToolbox/Phi/phi_star_Gauss.m',
                        'lib/PhiToolbox/Phi/phi_G_Gauss.m',
                        'lib/PhiToolbox/Phi/phi_G_Gauss_AL.m',
                        'lib/PhiToolbox/Phi/phi_G_Gauss_LBFGS.m',
                        'lib/PhiToolbox/Phi/phi_comp_probs.m',
                        'lib/PhiToolbox/Phi/phi_Gauss.m',
                        'lib/PhiToolbox/utility/I_s_I_s_d.m',
                        'lib/PhiToolbox/utility/data_to_probs.m',
                        'lib/PhiToolbox/utility/Gauss/Cov_comp.m',
                        'lib/PhiToolbox/utility/Gauss/Cov_cond.m',
                        'lib/PhiToolbox/utility/Gauss/H_gauss.m',
                        'lib/PhiToolbox/utility/Gauss/logdet.m',
                        'data/cml.npy',
                        'data/forex.npy',
                        'data/standard_normal.npy',
                        'data/cml7.npy']},
    include_package_data=True,
    version='0.4.2',
    description='Library for pairwise analysis of time series data.',
    author='Oliver M. Cliff',
    author_email='oliver.m.cliff@gmail.com',
    url='https://github.com/DynamicsAndNeuralSystems/pyspi',
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 1 - Planning",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Environment :: Console",
        "Environment :: Other Environment",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    install_requires=install_requires,
    extras_require={'testing': testing_extras}
)

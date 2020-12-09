from setuptools import setup, find_packages

# http://www.diveintopython3.net/packaging.html
# https://pypi.python.org/pypi?:action=list_classifiers

with open('README.md') as file:
    long_description = file.read()

setup(
    name='pynats',
    packages=find_packages(),
    #scripts=['bin/script1','bin/script2'],
    package_data={'': ['config.yaml','lib/jidt/infodynamics.jar']},
    include_package_data=True,
    version='0.0.12',
    description='Network analysis for time series',
    author='Oliver M. Cliff',
    author_email='oliver.m.cliff@gmail.com',
    url='https://github.com/olivercliff/pynats',
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
    install_requires=['pytest',
                        'numpy>=1.19.2',
                        'statsmodels>=0.12.0',
                        'pyyaml>=5.3.1',
                        'seaborn>=0.11.0',
                        'tqdm>=4.50.2',
                        'nilearn>=0.6.2',
                        'nitime==0.6',
                        'hyppo>=0.1.3',
                        'pyEDM>=1.0.3.2',
                        'jpype1>=1.2.0',
                        'sktime>=0.4.3',
                        'dill>=0.3.2',
                        'spectral-connectivity>=0.2.4.dev0',
                        'PyWavelets>=0.2.2',
                        'umap-learn>=0.4.6',
                        'torch>=1.7.0']
                 
)

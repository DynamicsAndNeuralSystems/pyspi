from setuptools import setup

# http://www.diveintopython3.net/packaging.html
# https://pypi.python.org/pypi?:action=list_classifiers

with open('README.md') as file:
    long_description = file.read()

setup(
    name='mtsda',
    packages=['mtsda'],
    #scripts=['bin/script1','bin/script2'],
    include_package_data=True,
    version='0.0.1',
    description='Multivariate time-series dependency analysis',
    author='Oliver M. Cliff, Ben D. Fulcher',
    author_email='oliver.m.cliff@gmail.com',
    url='https://github.com/olivercliff/mtsda',
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
    #install_requires=['pytest',...]
)

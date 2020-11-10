"""
    ScientificColourMaps5

    Usage
    -----
    import ScientificColourMaps5 as SCM5
    plt.imshow(data, cmap=SCM5.berlin)

    Available colourmaps
    ---------------------
    acton, bamako, batlow, berlin, bilbao, broc, buda, cork, davos, devon,
    grayC, hawaii, imola, lajolla, lapaz, lisbon, nuuk, oleron, oslo, roma,
    tofino, tokyo, turku, vik
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

folder = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

__all__ = {'acton', 'bamako', 'batlow', 'berlin', 'bilbao', 'broc', 'buda',
           'cork', 'davos', 'devon', 'grayC', 'hawaii', 'imola', 'lajolla',
           'lapaz', 'lisbon', 'nuuk', 'oleron', 'oslo', 'roma', 'tofino',
           'tokyo', 'turku', 'vik'}

for name in __all__:
    file = os.path.join(folder, name, name + '.txt')
    cm_data = np.loadtxt(file)
    cm = LinearSegmentedColormap.from_list(name, cm_data)
    plt.register_cmap(name=name, cmap=cm)
    plt.register_cmap(name=name + '_r', cmap=cm.reversed())

    # OMC: include the reversed colourmaps
    # vars()[name+'_r'] = LinearSegmentedColormap.from_list(name+'_r', np.flipud(cm_data))

    catfile = os.path.join(folder, name, 'CategoricalPalettes', name + 'S.txt')
    cm_data = np.loadtxt(file)
    cm = ListedColormap(cm_data,name+'S')
    plt.register_cmap(name=name+'S', cmap=cm)
import json
import numpy as np

from .models import Bicluster, Biclustering

"""
    biclustlib: A Python library of biclustering algorithms and evaluation measures.
    Copyright (C) 2017  Victor Alexandre Padilha

    This file is part of biclustlib.

    biclustlib is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    biclustlib is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


def save_biclusterings(b, file_path):
    """Dumps biclusterings to a file using the json module.

    Parameters
    ----------
    b : biclustlib.models.Biclustering or list
        A Biclustering instance or a list of Biclustering instances that will be saved.

    file_path : str
        The path of the file where the Biclustering instance or list of Biclustering instances will be saved.

    extension : str, default: 'json'
        The file extension to be used.
    """
    if not file_path.endswith('.json'):
        file_path += '.json'

    with open(file_path, 'w') as f:
        json.dump(b, f, default=_biclustering_to_dict)

def load_biclusterings(file_path):
    """Load biclusterings from a json file.

    Parameters
    ----------
    file_path : str
        The path of the file where the Biclustering instance or list of Biclustering instances are stored.
    """
    with open(file_path, 'r') as f:
        biclusterings = json.load(f, object_hook=_dict_to_biclustering)
    return biclusterings

def _biclustering_to_dict(bic):
    d = {'__class__' : bic.__class__.__name__, '__module__' : bic.__module__}

    try:
        d['biclusters'] =  [(list(map(int, b.rows)), list(map(int, b.cols)), b.data.tolist()) for b in bic.biclusters]
    except AttributeError: # for old compatibility
        d['biclusters'] =  [(list(map(int, b.rows)), list(map(int, b.cols))) for b in bic.biclusters]

    return d

def _dict_to_biclustering(bic_dict):
    try:
        biclust =  Biclustering([Bicluster(np.array(rows, np.int), np.array(cols, np.int), np.array(data, np.double)) for rows, cols, data in bic_dict['biclusters']])
    except ValueError:
        biclust =  Biclustering([Bicluster(np.array(rows, np.int), np.array(cols, np.int)) for rows, cols in bic_dict['biclusters']])
    return biclust
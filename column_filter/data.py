# -*- coding: utf-8 -*-
"""Download example data from my google drive."""

import os
import gdown

__all__ = ['download_data']


def download_data(dir_out):
    """Download example data.
    
    This function downloads example data from my personal google drive [1]_ 
    which enables you to directly test the code. Data is taken from a human 7 T 
    fMRI study in which ocular dominance columns in the primary visual cortex 
    were mapped using a high-resolution GE-EPI protocol. Further information
    can be found in the readme which is stored with the other data files.

    Parameters
    ----------
    dir_out : str
        Directory in which downloaded files are stored.

    Raises
    ------
    FileExistsError
        If `target` file location already exists.

    Returns
    -------
    dict
        Dictionary with keys pointing to the location of the downloaded files.

        * cortex : path to cortical surface mesh.
        * inflated : path to inflated cortical surface mesh
        * curv : path to curvature file
        * contrast : path to contrast overlay
        * label : path to V1 label file 
    
    References
    ----------
    .. [1] https://drive.google.com/drive/folders/1mXjr_fBtdl3xL0WCOwJyvgD3HTmks15y?usp=sharing

    """

    # make output folder
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    
    # base url to google drive folder
    url = 'https://drive.google.com/uc?export=download&id='
    
    # file IDs of source files
    file_id = ['1IbtTxHkWVhRPcRhYimEQUYaHHCIdC7FF',
               '1jLe5AzCONDzsMTnwBKotfY7UUBqhKoZv',
               '19B7e20UfQoCj4J7AJfGOAL6vTbJSFD2G',
               '130ftc_6VY2CTYUdDD3QSU7PaEShkEIDa',
               '19vKzT75Cnf6WSUW2kDpoWDY4LoRgjPbE',
               '132WHckZqhGq70LvivhC1XFypcDkNR1Nh',
               ]
    
    # file names of output files
    filename = ['lh.cortex',
                'lh.cortex_inflated',
                'lh.curv',
                'lh.contrast.mgh',
                'lh.v1.label',
                'README.md'
                ]
    
    file_sources = [url+id_ for id_ in file_id]
    file_targets = [os.path.join(dir_out, file) for file in filename]
    
    for source, target in zip(file_sources, file_targets):
        
        # check if file already exists
        if os.path.exists(target):
            raise FileExistsError("The file "+target+" already exists!")
        else:
            gdown.download(source, target, quiet=False)
    
    return {'cortex': file_targets[0],
            'inflated': file_targets[1],
            'curv': file_targets[2],
            'contrast': file_targets[3],
            'label': file_targets[4]}

nmrpca
======

Principal Component Analysis for Nuclear Magnetic Resonance Spectrometry data

A collection of tools for Principal Component Analysis (PCA) on NMR data.
The focus is on _in situ_ or _in operando_ NMR and potentially on T1 / T2 fitting.

### Basic usage ###

    import sklearn.decomposition
    import nmrpca
    
    # Get complex data ...
    
    flat_data = nmrpca.nmr_flatten(complex_data)
    pca = sklearn.decomposition.PCA()
    coefficients = pca.fit_transform(flat_data)
    components = nmrpca.nmr_rebuild(pca.components_)
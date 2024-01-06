linex2metaspace documentation
=============================

This python package provides functions to rank (lipid) annotations in imaging Mass Spectrometry data
based on prior knowledge. The package is fully compatible and 
designed to work with `METASPACE`_ data.
The package uses as input data from `METASPACE`_, downloaded through the `METASPACE python client`_
or the `metaspace-converter`_ package.

For a detailed explanation of the approach, please have a look at the publication: **TODO**

In short, it uses dataset-specific lipid metabolic networks (computed by the `LINEX2`_ approach) to
rank annotation candidates by their network connectivity as an 
approximation for biochemical likelihood.

Have a look at the provided examples for a detailed description of the package usage.


.. toctree::
   :maxdepth: 1
   :caption: Example workflows

   examples/python_client_annotation


.. toctree::
   :maxdepth: 1
   :caption: API reference:
   
   api
   

.. _METASPACE: https://metaspace2020.eu/
.. _METASPACE python client: https://github.com/metaspace2020/metaspace/tree/master/metaspace/python-client
.. _metaspace-converter: https://metaspace2020.github.io/metaspace-converter/index.html
.. _LINEX2: https://exbio.wzw.tum.de/linex/

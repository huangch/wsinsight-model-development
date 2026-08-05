Installing wsinsight-train
===========================

Recommended setup:

.. code-block:: bash

   bash conda-setup.sh -n wsitrain -r
   conda activate wsitrain
   export CELLVIT_ROOT=/path/to/CellViT-plus-plus

Smoke test:

.. code-block:: bash

   wsitrain --version
   wsitrain check --input /path/to/cohort --tissue pantissue

Training example:

.. code-block:: bash

   wsitrain run --input /path/to/cohort --tissue breast --task sthelar_full

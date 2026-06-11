Neuromorphic EVolutionary Optimisation (NEVO)
==============================================

**NEVO** uses a Nengo-simulated Basal Ganglia (BG) circuit to adaptively select
optimisation operators at runtime. The search loop is as follows:

   Extract state features → Compute operator utilities →
   BG selection → Generate population →
   Evaluate → Update memory & TD weights

As a general overview, the loop is represented in the following diagram:

.. image:: /images/cbtl-simple.png
   :align: center
   :alt: Loop diagram

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started
   examples

.. toctree::
   :maxdepth: 1
   :caption: User Guides

   ARCHITECTURE
   NEUROMORPHIC_ENSEMBLES
   td_learning_guide
   QUICK_REFERENCE_NEUROMORPHIC

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Project

   changelog_link
   contributing


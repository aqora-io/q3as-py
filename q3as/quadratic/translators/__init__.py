# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Quadratic program translators (:mod:`qiskit_optimization.translators`)
======================================================================

.. currentmodule:: qiskit_optimization.translators

Translators between :class:`~qiskit_optimization.problems.QuadraticProgram` and
other optimization models or other objects.

Translators
----------------------
.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   from_ising
   to_ising
"""

from .ising import from_ising, to_ising

__all__ = [
    "from_ising",
    "to_ising",
]

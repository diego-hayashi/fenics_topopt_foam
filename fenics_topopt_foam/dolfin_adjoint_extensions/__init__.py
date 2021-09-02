"""
Some useful functions for using with dolfin-adjoint.
"""

#
# Copyright (C) 2020-2021 Diego Hayashi Alonso
#
# This file is part of FEniCS TopOpt Foam.
# 
# FEniCS TopOpt Foam is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FEniCS TopOpt Foam is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with FEniCS TopOpt Foam. If not, see <https://www.gnu.org/licenses/>.
#

# Some useful functions/classes to import when using:
 # "from fenics_topopt_foam_setups.dolfin_adjoint_solvers import * "

__all__ = [
	'UncoupledLinearVariationalSolver',
	'UncoupledNonlinearVariationalSolver',
	'getWallDistanceAndNormalVectorFromDolfinAdjoint',
]

from .UncoupledLinearVariationalSolver import UncoupledLinearVariationalSolver
from .UncoupledNonlinearVariationalSolver import UncoupledNonlinearVariationalSolver
from .getWallDistanceAndNormalVectorFromDolfinAdjoint import getWallDistanceAndNormalVectorFromDolfinAdjoint



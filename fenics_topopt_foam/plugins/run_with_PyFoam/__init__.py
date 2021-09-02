"""
Run with PyFoam.

Description:
	Call '[FoamSolver].solve' with 'run_mode' specifying a PyFoam utility name!
	Just don't forget to install it:
		(1) Enter the OpenFOAM environment
		(2) Install PyFoam with pip:
			$ pip install --user PyFoam

Example:
	Call '[FoamSolver].solve' with 'run_mode' specifying a PyFoam utility name:

		[FoamSolver].solve([...], run_mode = 'pyFoamRunner.py')

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

from .run_with_PyFoam import *


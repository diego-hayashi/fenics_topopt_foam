################################################################################
#                               run_with_PyFoam                                #
################################################################################

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

############################### Python libraries ###############################

# NumPy
import numpy as np

############################# Project libraries ################################
# Just be careful with cyclic imports! Don't import a function that imports 
 # the overloaded function. Anyway, you can use the input parameter 
 # 'module_being_overloaded' for these cases.

# FEniCS TopOpt Foam folder
from ... import __fenics_topopt_foam_folder__

# Utilities
from ...utils import utils

# FEniCS utilities
from ...utils import utils_fenics

# FEniCS+MPI utilities
from ...utils import utils_fenics_mpi

################################## Plugin setup ################################

# Plugin priority (i.e., which plugin loads first? Higher values mean higher priority!)
plugin_priority = 1

# Plugin overloads (i.e., a list with all overloads)
plugin_overloads = []

################################################################################
################################## FoamSolver ##################################
################################################################################

############################## FoamSolver.__solve ##############################

def FoamSolver_method___solve(self, _old__solve, module_being_overloaded, *args, **kwargs):
	"""
	Run OpenFOAM with PyFoam
	 https://openfoamwiki.net/index.php/Contrib/PyFoam
	
	 * Before using PyFoam post-processing, you need to:
		(1) Enter the OpenFOAM environment
		(2) Install PyFoam with pip:
			$ pip install --user PyFoam

	 * Some PyFoam utility names are:
		run_mode = 'pyFoamRunner.py'
		run_mode = 'pyFoamSteadyRunner.py'
		run_mode = 'pyFoamPlotRunner.py'
		run_mode = 'pyFoamMeshUtilityRunner.py'
		run_mode = 'pyFoamPotentialRunner.py'
		run_mode = 'pyFoamRunAtMultipleTimes.py'
		run_mode = 'pyFoamRunParameterVariation.py'

	"""

	run_mode_orig = kwargs['run_mode']

	if (type(run_mode_orig).__name__ == 'str') and ('pyFoam' in run_mode_orig):

		def run_mode(foam_run_command):

			# Wait for everyone!
			if utils_fenics_mpi.runningInParallel():
				utils_fenics_mpi.waitProcessors()

			if utils_fenics_mpi.runningInSerialOrFirstProcessor():
				with utils_fenics_mpi.first_processor_lock():
					bash_response = utils.run_command_in_shell('echo -n $(type %s)' %(run_mode), mode = 'save all prints to variable', indent_print = True)
			else:
				bash_response = None

			# Wait for everyone!
			if utils_fenics_mpi.runningInParallel():
				utils_fenics_mpi.waitProcessors()

			if utils_fenics_mpi.runningInParallel():
				bash_response = utils_fenics_mpi.broadcastToAll(bash_response)

			if "not found" in bash_response:
				raise ValueError(" ❌ ERROR: run_mode == '%s' is not available!" %(run_mode))
			else:
				runner_utility_name = run_mode

			foam_run_command = '$s %s' %(runner_utility_name, foam_run_command)

			utils.customPrint("\n 🌊 Solving the problem with OpenFOAM coupled with PyFoam (%s) post-processing..." %(runner_utility_name))

		kwargs['run_mode'] = run_mode

	return _old__solve(self, *args, **kwargs)

# ---------------------------------------------------------------------------- #

plugin_overloads += [
	{ # Plugin overload dictionary (i.e., a dictionary with the overload configurations)

		# File with the variable to overload
		'file' : '%s/solver/FoamSolver.py' %(__fenics_topopt_foam_folder__),

		# Variable to overload
		'variable to overload' : {

			# Type of overloading operation
			'type' : 'overload method from class',

				# 'overload method from class'
				# 'new method for class'
				# 'new self.variable for class'
				# 'new class'

				# 'overload function'
				# 'new function'

			'overload method from class' : {
				'class name' : 'FoamSolver',
				'method name' : '__solve',
				'overloader method' : FoamSolver_method___solve,
				'add to method docstring' : 'solve',
			}
		}
	}
]

# ---------------------------------------------------------------------------- #



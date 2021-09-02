################################################################################
#                           generateMeshInOpenFOAM                             #
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

############################# Project libraries ################################

# Utilities
from ..utils import utils

########################### generateMeshInOpenFOAM #############################

@utils.only_the_first_processor_executes()
def generateMeshInOpenFOAM(problem_folder, parameters_for_mesh_generation):
	"""
	Generates a mesh in OpenFOAM.
	"""

	if parameters_for_mesh_generation['type'] == 'blockMesh':
		utils.customPrint(" 🌀 Generating mesh by using the '%s' OpenFOAM utility..." %(parameters_for_mesh_generation['type']))
		utils.run_command_in_shell('blockMesh -case %s' %(problem_folder), mode = 'print directly to terminal', indent_print = True)
	else:
		raise ValueError(" ❌ ERROR: parameters_for_mesh_generation['type'] == '%s' is not defined!" %(parameters_for_mesh_generation['type']))


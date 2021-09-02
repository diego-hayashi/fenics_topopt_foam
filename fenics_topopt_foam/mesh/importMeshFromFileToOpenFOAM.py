################################################################################
#                         importMeshFromFileToOpenFOAM                         #
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

######################## importMeshFromFileToOpenFOAM ##########################

@utils.only_the_first_processor_executes()
def importMeshFromFileToOpenFOAM(mesh_file_location, problem_folder, mesh_file_type = 'Ansys Fluent ASCII 2D'):
	"""
	Import mesh to OpenFOAM.
	Check all available mesh conversion utilities in [$FOAM_APP]/utilities/mesh/conversion
	https://www.openfoam.com/documentation/user-guide/mesh-conversion.php
	"""

	if mesh_file_type == 'Gmsh':
		utility_to_use = 'gmshToFoam'

	elif mesh_file_type == 'Ansys Fluent ASCII 2D':
		utility_to_use = 'fluentMeshToFoam'

	elif mesh_file_type == 'Ansys Fluent ASCII 3D':
		utility_to_use = 'fluent3DMeshToFoam'

	elif mesh_file_type == 'CFX4':
		utility_to_use = 'cfx4ToFoam'

	elif mesh_file_type == 'VTK': # [$FOAM_APP]/utilities/mesh/conversion/vtkUnstructuredToFoam/
					# https://openfoam.org/release/2-2-0/pre-processing-macros-patch-groups/
		utility_to_use = 'vtkUnstructuredToFoam'

	else:
		raise ValueError(" ❌ ERROR: mesh_file_type == '%s' is not defined here yet! Check [$FOAM_APP]/utilities/mesh/conversion if it is available in OpenFOAM!" %(unit_name))

	utils.customPrint(" 🌀 Importing mesh file '%s' (type: %s) to OpenFOAM..." %(mesh_file_location, mesh_file_type))
	utils.run_command_in_shell('%s -case %s %s' %(utility_to_use, problem_folder, mesh_file_location), mode = 'print directly to terminal', indent_print = True)


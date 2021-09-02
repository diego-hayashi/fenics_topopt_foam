################################################################################
#                               exportMesh                                     #
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

################################# exportMesh ###################################

@utils.only_the_first_processor_executes()
def exportMesh(problem_folder, mesh_file_type = 'Ansys Fluent ASCII 2D', time_step_name = 'last'):
	"""
	Export mesh from OpenFOAM.
	Check all available mesh conversion utilities in [$FOAM_APP]/utilities/mesh/conversion
	https://www.openfoam.com/documentation/user-guide/mesh-conversion.php
	https://cfd.direct/openfoam/user-guide/v6-mesh-conversion/
	"""

	if mesh_file_type == 'Ansys Fluent':
		utility_to_use = 'foamMeshToFluent'

		if time_step_name == 'last':
			additional_options = '-latestTime'
		else:
			additional_options = '-time %s' %(time_step_name)

	elif mesh_file_type == 'VTK': # [$FOAM_APP]/utilities/postProcessing/dataConversion/foamToVTK
					# If there are results available, it also generates the VTK files of the results.
		utility_to_use = 'foamToVTK'

		if time_step_name == 'last':
			additional_options = '-ascii -latestTime'
		else:
			additional_options = '-ascii -time %s' %(time_step_name)

	else:
		raise ValueError(" ❌ ERROR: mesh_file_type == '%s' is not defined here yet! Check OpenFOAM-7/applications/utilities/mesh/conversion if it is available in OpenFOAM!" %(mesh_file_type))

	utils.customPrint(" 🌀 Exporting OpenFOAM mesh to %s..." %(mesh_file_type))
	utils.run_command_in_shell('%s -case %s %s' %(utility_to_use, problem_folder, additional_options), mode = 'print directly to terminal', indent_print = True)



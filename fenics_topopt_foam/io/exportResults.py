################################################################################
#                              exportResults                                   #
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

############################## exportResults ###################################

@utils.only_the_first_processor_executes()
def exportResults(problem_folder, file_type = 'VTK', time_step_name = 'last', more_options = ""):
	"""
	Export OpenFOAM results.
	"""

	if file_type == 'VTK': # [$FOAM_APP]/utilities/postProcessing/dataConversion/foamToVTK
		utils.customPrint(" 🌀 Exporting results to VTK... (* These are the results that are plotted in ParaView if the command \"paraFoam -case %s\" is run)" %(problem_folder))
		utility_to_use = 'foamToVTK'

		assert '-time ' not in more_options
		assert '-latestTime' not in more_options

		if time_step_name == 'last':
			additional_options = '-ascii -latestTime'
		else:
			additional_options = '-ascii -time %s' %(time_step_name)
		additional_options += more_options

	else:
		raise ValueError(" ❌ ERROR: file_type == '%s' is not defined here!" %(file_type))

	utils.run_command_in_shell('%s -case %s %s' %(utility_to_use, problem_folder, additional_options), mode = 'print directly to terminal', indent_print = True)


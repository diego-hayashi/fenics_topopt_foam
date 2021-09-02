################################################################################
#                         createProblemFolderFromMesh                          #
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

# Utilities
from ..utils import utils

# FEniCS utilities
from ..utils import utils_fenics

# FEniCS+MPI utilities
from ..utils import utils_fenics_mpi

# FEniCS TopOpt Foam folder
from .. import __fenics_topopt_foam_folder__

######################### createProblemFolderFromMesh ##########################

def createProblemFolderFromMesh(problem_folder, polyMesh_folder_to_copy, optimize_mesh = False, check_mesh = False, time_step_name = '0'):
	"""
	Create a problem folder from a given 'polyMesh' folder.

	* It creates a dummy problem folder from a given mesh file
	  in such a way that you can run "createFEniCSMeshFromOpenFOAM".
	"""

	# Check folders
	utils.customPrint(" 🌀 Checking necessary folders...")
	utils.createFolderIfItDoesntExist("%s" %(problem_folder))
	utils.createFolderIfItDoesntExist("%s/%s" %(problem_folder, time_step_name))
	utils.createFolderIfItDoesntExist("%s/constant" %(problem_folder))

	# Copy polyMesh to problem folder
	utils.customPrint(" 🌀 Copying polyMesh to problem folder...")
	folder_polyMesh = '%s/constant/polyMesh' %(problem_folder)
	utils.removeFolderIfItExists(folder_polyMesh)
	utils.copyFolder(polyMesh_folder_to_copy, folder_polyMesh, overwrite = False)

	# Copy dummy files for the 'system' folder, in such a way that you may run "createFEniCSMeshFromOpenFOAM"
	utils.customPrint(" 🌀 Copying some dummy files (based in the 'pitzDaily' OpenFOAM tutorial) for the 'system' folder..")
	folder_system = '%s/system' %(problem_folder)
	utils.removeFolderIfItExists(folder_system)
	utils.copyFolder('%s/mesh/dummy_files/pitzDaily_system_folder' %(__fenics_topopt_foam_folder__), folder_system, overwrite = False)

	# Optimize mesh
	if optimize_mesh == True:

		# Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

		if utils_fenics_mpi.runningInSerialOrFirstProcessor():
			with utils_fenics_mpi.first_processor_lock():
				utils.run_command_in_shell("renumberMesh -case %s -overwrite" %(problem_folder), mode = 'print directly to terminal', indent_print = True)

		# Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

	# Check mesh
	if check_mesh == True:

		# Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

		if utils_fenics_mpi.runningInSerialOrFirstProcessor():
			with utils_fenics_mpi.first_processor_lock():
				utils.run_command_in_shell("checkMesh -case %s -constant -time constant" %(problem_folder), mode = 'print directly to terminal', indent_print = True)

		# Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()


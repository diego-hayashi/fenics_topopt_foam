################################################################################
#                                 FoamReader                                   #
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

############################## Python libraries ################################

# os
import os

############################# Project libraries ################################

# FEniCS TopOpt Foam folder
from .. import __fenics_topopt_foam_folder__

# Types
from ..types.FoamMesh import FoamMesh
from ..types.FoamVector import FoamVector
from ..types.FoamProperty import FoamProperty
from ..types.FoamConfiguration import FoamConfiguration

# Utilities
from ..utils import utils

# OpenFOAM information
from ..utils import foam_information

################################## FoamReader ##################################

class FoamReader():
	"""
	Reader for reading setup from OpenFOAM files to Python dictionaries.
	"""

	def __init__(self, problem_folder, foam_mesh = None):

		# Problem folder
		self.problem_folder = problem_folder

		# Folders
		self.properties_folder = "%s/constant" %(self.problem_folder)
		self.configuration_folder = "%s/system" %(self.problem_folder)

		# FoamMesh
		self.foam_mesh = foam_mesh

	######################## _convertOpenFOAMtoPython ######################

	def _convertOpenFOAMtoPython(self, data_file_name):
		"""
		Convert an OpenFOAM dictionary file to Python.
		"""

		# Convert the FoamFile to a Python dictionary
		utils.run_command_in_shell("sh '%s/io/sh/openfoamtopy.sh' '%s'" %(__fenics_topopt_foam_folder__, data_file_name), mode = 'print directly to terminal', include_delimiters_for_print_to_terminal = False, indent_print = True)

	#################### _readDataFromFoamFile #############################

	@utils.only_the_first_processor_executes(broadcast_result = True)
	def _readDataFromFoamFile(self, data_file_name, remove_python_file = True):
		"""
		Read the OpenFOAM dictionary in Python.
		"""

		# Convert the FoamFile to a Python dictionary
		self._convertOpenFOAMtoPython(data_file_name)

		# Load the Python dictionary
		module = utils.loadPythonFile("%s.py" %(data_file_name), force_reload = True)
		data = module.data

		# Remove the Python file
		if remove_python_file == True:
			utils.removeFileIfItExists("%s.py" %(data_file_name), mpi_wait_for_everyone = False)
			data_file_folder = os.path.dirname(data_file_name)
			utils.removeFolderIfItExists("%s/__pycache__" %(data_file_folder), mpi_wait_for_everyone = False)

		return data

	########################################################################
	############################ FoamMesh ##################################
	########################################################################

	####################### readMeshToFoamMesh #############################

	def readMeshToFoamMesh(self, mpi_broadcast_result = True, mpi_wait_for_everyone = True):
		"""
		Reads the mesh to a FoamMesh.
		"""

		data_names = ['boundary', 'faces', 'neighbour', 'owner', 'points']

		data = {}
		for data_name in data_names:
			new_data = self.readMeshData(data_name, mpi_broadcast_result = mpi_broadcast_result, mpi_wait_for_everyone = mpi_wait_for_everyone)
			data[data_name] = new_data

		# We may need to compute some more data later (i.e, on demand)
		data['FoamReader'] = self
		data['problem_folder'] = self.problem_folder

		foam_mesh = FoamMesh(data)
		self.foam_mesh = foam_mesh

		return foam_mesh

	######################## readMeshBoundary ##############################

	def readMeshData(self, data_name, remove_python_file = True, mpi_broadcast_result = True, mpi_wait_for_everyone = True):
		"""
		Reads some mesh data.
		"""

		# FoamFile
		mesh_data_file = "%s/polyMesh/%s" %(self.properties_folder, data_name)

		# Read the FoamFile
		data = self._readDataFromFoamFile(mesh_data_file, remove_python_file = remove_python_file, mpi_broadcast_result = mpi_broadcast_result, mpi_wait_for_everyone = mpi_wait_for_everyone)

		return data

	########################################################################
	######################### FoamConfiguration ############################
	########################################################################

	############### readConfigurationToFoamConfiguration ###################

	def readConfigurationToFoamConfiguration(self, configuration_name, remove_python_file = True):
		"""
		Reads a variable to a FoamConfiguration.
		"""

		data = {}

		utils.customPrint(" ❗ Reading configuration from file is not implemented! Initializing empty configuration \"%s\"" %(configuration_name))

		return FoamConfiguration(data, name = configuration_name)

	########################################################################
	############################ FoamProperty ##############################
	########################################################################

	#################### readPropertyToFoamProperty ########################

	def readPropertyToFoamProperty(self, property_name, remove_python_file = True, mpi_broadcast_result = True):
		"""
		Reads a variable to a FoamProperty.
		"""

		# FoamFile
		property_file = "%s/%s" %(self.properties_folder, property_name)

		# Read the FoamFile
		data = self._readDataFromFoamFile(property_file, remove_python_file = remove_python_file, mpi_broadcast_result = mpi_broadcast_result)

		return FoamProperty(data)

	########################################################################
	############################## FoamVector ##############################
	########################################################################

	###################### readVariableToFoamVector ########################

	def readVariableToFoamVector(self, variable_name, time_step_name = 'last', remove_python_file = True, mpi_broadcast_result = True):
		"""
		Reads a variable to a FoamVector.
		"""

		# Check if you want the last timestep
		if time_step_name == 'last':
			sorted_number_folders = utils.findFoamVariableFolders(self.problem_folder)
			time_step_name = sorted_number_folders[len(sorted_number_folders) - 1]

		if utils.checkIfFileExists("%s/%s/%s" %(self.problem_folder, time_step_name, variable_name)):
			pass
		else:
			utils.customPrint("❗Variable file for '%s' in time step '%s' does not exist (* This is true for variables not changed in the solver.). Loading the value at time step '0'..." %(variable_name, time_step_name))
			time_step_name = '0'

		# FoamFile
		variable_file = "%s/%s/%s" %(self.problem_folder, time_step_name, variable_name)

		# Read the FoamFile
		data = self._readDataFromFoamFile(variable_file, remove_python_file = remove_python_file, mpi_broadcast_result = mpi_broadcast_result)

		# FoamMesh
		if type(self.foam_mesh).__name__ == 'NoneType':
			raise ValueError(" ❌ ERROR: FoamMesh is not defined!")
		foam_mesh = self.foam_mesh

		return FoamVector(foam_mesh, data)


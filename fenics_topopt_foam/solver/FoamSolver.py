################################################################################
#                                  FoamSolver                                  #
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

import os
import numpy as np

from datetime import datetime

try:
	import matplotlib

	# Check current Matplotlib backend
	current_backend = matplotlib.rcParams['backend']

	# Check if it is a non-interactive backend
	 # https://stackoverflow.com/questions/45993879/matplot-lib-fatal-io-error-25-inappropriate-ioctl-for-device-on-x-server-loc
	 # https://matplotlib.org/faq/howto_faq.html
	 # * This is because there is (in this code) a scheme for plotting with Matplotlib in a fork, which requires this.
	  # * Otherwise, it will probably return an error: "  XIO:  fatal IO error 25 (Inappropriate ioctl for device) on X server ":0"  "
	if current_backend.lower() not in ['agg', 'ps', 'pdf', 'svg', 'cairo']:
		print(""" ❌ ATTENTION: Your backend is '%s', which is probably non-interactive. A non-interactive backend is required for running Matplotlib for plotting the residuals in a fork! 
Please, add something like:

import matplotlib
matplotlib.use('agg') # 'agg' is a non-interactive backend

in the beginning of your code, before performing any Matplotlib import in your code 
(because, after other imports (such as matplotlib.pyplot), Matplotlib "locks" the backend choice, and it can not be 
changed anymore). The choice of the backend is not enforced in FEniCS TopOpt Foam, because 
the user may want to import and use Matplotlib before importing FEniCS TopOpt Foam.
""" %(current_backend))

	import matplotlib.pyplot as plt
except:
	print(" ❌ Matplotlib is not available in your intallation!")

############################# Project libraries ################################

# Mesh
from ..mesh.generateMeshInOpenFOAM import generateMeshInOpenFOAM
from ..mesh.importMeshFromFileToOpenFOAM import importMeshFromFileToOpenFOAM
from ..mesh.exportMesh import exportMesh

# I/O (Input/Output)
from ..io.FoamWriter import FoamWriter
from ..io.FoamReader import FoamReader
from ..io.FoamMeshWriter import FoamMeshWriter
from ..io.exportResults import exportResults

# Types
from ..types.FoamMesh import FoamMesh
from ..types.FoamVector import FoamVector
from ..types.FoamProperty import FoamProperty
from ..types.FoamConfiguration import FoamConfiguration

# Utilities
from ..utils import utils

# FEniCS+MPI utilities
from ..utils import utils_fenics_mpi

# OpenFOAM information
from ..utils import foam_information

# FEniCS TopOpt Foam folder
from .. import __fenics_topopt_foam_folder__

################################## FoamSolver ##################################

class FoamSolver():
	"""
	Solver structure for interfacing OpenFOAM with Python.
	"""

	def __init__(self, parameters = {}, python_write_precision = 6):

		utils.customPrint("\n 🌊 Creating FoamSolver...", mpi_wait_for_everyone = True)

		# Write precision
		self.python_write_precision = python_write_precision

		################## OpenFOAM verifications ######################

		# Initial OpenFOAM print
		self._initialFoamPrint()

		# Check if the custom OpenFOAM utilities are installed (if there is any needed beyond the default OpenFOAM utilities)
		if parameters.get('compile_modules_if_needed', True) == True:
			utils.checkCustomOpenFOAMUtilities()

		####################### Parameters #############################

		# Default parameters
		self.parameters = {

			#################### Domain type ########################
			'domain type' : '2D', 
				# '3D' = Consider a "3D" mesh
				# '2D' = Consider a "2D" mesh (i.e., include the necessary boundary conditions in the 3D mesh for it)
				# '2D axisymmetric' = Consider a "2D axisymmetric" mesh (i.e., include the necessary boundary conditions in the 3D mesh for it)

			############### Error on nonconvergence #################

			# Error on nonconvergence
			'error_on_nonconvergence' : True,
				# True
				# False

			# Error on nonconvergence when OpenFOAM fails, either by diverging or getting, for example, a floating-point error.
			'error_on_nonconvergence_failure' : True, 
				# True
				# False

			#################### Problem folder #####################
			# Folder with the OpenFOAM file structure and the complete definition of the problem
			'problem_folder' : "foam_problem", 

			######################## Solver #########################
			'solver' : {
				'type' : 'custom',
					# 'openfoam' = Use a solver that is already defined in the OpenFOAM environment. Check out: 
						# https://cfd.direct/openfoam/user-guide/v7-standard-solvers/
						# https://www.openfoam.com/documentation/user-guide/standard-solvers.php
					# 'custom' = Use a solver defined by the user. If the modification date of the code is more recent than the modification date of compilation, the solver is recompiled.
				'openfoam' : {
					'name' : 'simpleFoam', # Name of the solver in the OpenFOAM environment
				},
				'custom' : {
					'name' : 'customSimpleFoam',     # Name of the solver, which is also the name of the folder that contains it
					'location' : 'foam_cpp_solvers', # Folder that contains the solver
				},
			},

			######################## Mesh ###########################
			'mesh' : {
				'type' : 'OpenFOAM mesh',
					# 'OpenFOAM mesh' = Uses the mesh that is already prepared in the OpenFOAM format.
					# 'import from file' = Imports the mesh from a file.
					# 'generate mesh with OpenFOAM' = Uses the file in [problem_folder]/blockMeshDict to generate the mesh in OpenFOAM.
				'OpenFOAM mesh' : {
				},
				'import from file' : {
					'file_type' : 'Gmsh', 
						# 'Gmsh' = Mesh from Gmsh (.msh) (* For now (OpenFOAM 7.0), the format is Gmsh2, and the boundary information is not correctly recognized by the converter)
						# 'Ansys Fluent ASCII 2D' = Mesh from Ansys Fluent 2D mesh
						# 'Ansys Fluent ASCII 3D' = Mesh from Ansys Fluent 3D mesh
						# 'CFX4' = Mesh from Ansys CFX mesh
						# 'VTK' = Mesh from VTK file (* Does not contain boundary information)
					'file_location' : 'mesh-gmsh.msh',
				},
				'generate OpenFOAM mesh' : { 
					'type' : 'blockMesh',
						# 'blockMesh' = Generate from [problem_folder]/blockMeshDict (https://cfd.direct/openfoam/user-guide/v7-blockmesh/)
				}
			},

			################## Module compilation ###################
			# If custom modules should be compiled/recompiled when needed.
			'compile_modules_if_needed' : True,
				# True
				# False
		}

		# Set the new parameters selected by the user
		self.parameters.update(parameters)

		####################### Parameters check #######################

		# Remove trailing "/" of folder names, in case the user wrote "/" in the end
		self.parameters['problem_folder'] = utils.removeTrailingSlash(self.parameters['problem_folder'])
		self.parameters['solver']['openfoam']['name'] = utils.removeTrailingSlash(self.parameters['solver']['openfoam']['name'])
		self.parameters['solver']['custom']['name'] = utils.removeTrailingSlash(self.parameters['solver']['custom']['name'])
		self.parameters['solver']['custom']['location'] = utils.removeTrailingSlash(self.parameters['solver']['custom']['location'])

		#################### Location of the solver ####################

		if self.parameters['solver']['type'] == 'openfoam':

			# Location of the OpenFOAM solver
			self.location_solver = utils.getOpenFOAMSolverLocation(self.parameters['solver']['openfoam']['name'])

		elif self.parameters['solver']['type'] == 'custom':

			# Location of the custom solver
			assert self.parameters['solver']['custom']['location'] != ""
			self.location_solver = "%s/%s" % (self.parameters['solver']['custom']['location'], self.parameters['solver']['custom']['name'])

		else:
			raise ValueError(" ❌ ERROR: self.parameters['solver']['type'] == '%s' is not defined!" %(self.parameters['solver']['type']))

		self.solver_name = self.parameters['solver'][self.parameters['solver']['type']]['name']

		######### Compilation of the custom solver if necessary ########

		# Compile the solver (if needed)
		if self.parameters['solver']['type'] == 'custom' and self.parameters['compile_modules_if_needed'] == True:
			utils.compileModuleIfNeeded("%s/%s" %(self.parameters['solver']['custom']['location'], self.parameters['solver']['custom']['name']), force_recompile = False)

		############################# Mesh #############################

		self._prepareMeshForOpenFOAM()

		####################### ready_for_editing ######################

		self.ready_for_editing = False

		#### Flag that indicates if it is to run in parallel or not ####

		# Initially set not to run in parallel, because it requires
		 # an additional file for it
		self.RUN_IN_PARALLEL = False

		# Number of subdomains
		self.NUMBER_OF_SUBDOMAINS = None

		# Variable environment for reinitialization
		self.variable_environment_reinitialization_for_parallel = {}

	######################### _initialFoamPrint ############################

	def _initialFoamPrint(self):
		"""
		Sets up some OpenFOAM information for this class.
		"""

		utils.customPrint("""
────────────────────────────────────────────────────────────────────────────────
                             FEniCS TopOpt Foam
────────────────────────────────────────────────────────────────────────────────
  ═════════                 │ OpenFOAM implementation:  %s
  ╲╲      ╱  F ield         │ ─────────────────────────────────────────────────
   ╲╲    ╱   O peration     │ Version:                  %s 
    ╲╲  ╱    A nd           │ Installation directory:   %s
     ╲╲╱     M anipulation  │ Source code location:     %s
────────────────────────────────────────────────────────────────────────────────
""" % 		(
		foam_information._foam_implementation_,
		foam_information._foam_version_,
		foam_information._foam_inst_dir_,
		foam_information._foam_src_
		)
		)

	######################## _prepareMeshForOpenFOAM #######################

	def _prepareMeshForOpenFOAM(self, mesh_configs = {}, problem_folder = ""):
		"""
		Prepare the mesh for OpenFOAM.
		"""

		if len(mesh_configs) == 0:
			mesh_configs = self.parameters['mesh']

		if problem_folder == "":
			problem_folder = self.parameters['problem_folder']

		# [Parallel] Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

		if mesh_configs['type'] == 'OpenFOAM mesh': # Use the mesh that is already prepared in the OpenFOAM format
			utils.customPrint("\n 🌀 Using mesh already specified in OpenFOAM format from '%s/constant'..." %(problem_folder))

		elif mesh_configs['type'] == 'import from file': # Imports the mesh from a file

			importMeshFromFileToOpenFOAM(mesh_configs['import from file']['file_location'], problem_folder, mesh_file_type = mesh_configs['import from file']['file_type'])

		elif mesh_configs['type'] == 'generate mesh with OpenFOAM': # Use the file in [problem_folder]/blockMeshDict to generate the mesh in OpenFOAM
			generateMeshInOpenFOAM(mesh_configs['generate OpenFOAM mesh'])

		else:
			raise ValueError(" ❌ ERROR: mesh_configs['type'] == '%s' is not defined!" %(mesh_configs['type']))

		# Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

		# Create cell zones from topoSetDict
		 # https://openfoamwiki.net/index.php/TopoSet
		if 'topoSetDict' in utils.getNamesOfFilesInFolder("%s/system" %(problem_folder)):
			if utils_fenics_mpi.runningInSerialOrFirstProcessor():
				with utils_fenics_mpi.first_processor_lock():
					utils.customPrint("\n 🌀 Running topoSet (maybe for setting cellZones in the mesh)...")
					utils.run_command_in_shell("topoSet -case '%s'" %(problem_folder), mode = 'print directly to terminal', indent_print = True)

		# Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

	######################### prepareForEditing ############################

	def prepareForEditing(self, time_step_name = '0', keep_previous_setup = False, keep_previous_mesh = False):
		"""
		Prepare FoamSolver for editing.
		"""

		utils.customPrint("\n 🌊 Preparing FoamSolver for editing...")

		######################### Read/write ###########################

		# Create a reader for the OpenFOAM files
		if 'foam_reader' not in self.__dict__:
			self.foam_reader = FoamReader(self.parameters['problem_folder'])

		# Create a writer for the OpenFOAM files
		if 'foam_writer' not in self.__dict__:
			self.foam_writer = FoamWriter(self.parameters['problem_folder'], python_write_precision = self.python_write_precision)

		########################### FoamMesh ###########################

		# Read the mesh from OpenFOAM
		if ('foam_mesh' not in self.__dict__) or (keep_previous_mesh == False):
			self.foam_mesh = self._prepareFoamMesh()

		########################## FoamVectors #########################

		# Read the variables from OpenFOAM
		 # These are the new values!
		self.foam_vectors = self._prepareFoamVectors(time_step_name)

		########################## FoamProperties ######################

		# Read the properties from OpenFOAM
		if ('foam_properties' not in self.__dict__) or (keep_previous_setup == False):
			self.foam_properties = self._prepareFoamProperties()

		####################### FoamConfigurations #####################

		# Read the configurations from OpenFOAM
		if ('foam_configurations' not in self.__dict__) or (keep_previous_setup == False):
			self.foam_configurations = self._prepareFoamConfigurations()
				# * Remember that 'foam_reader.readConfigurationToFoamConfiguration' 
				 #  returns an empty configuration object, because reading the 
				 #  corresponding OpenFOAM file is not implemented (at least for now).

		####################### ready_for_editing ######################

		self.ready_for_editing = True

	def getFoamVectors(self):
		"""
		Returns the FoamVectors.
		"""

		if self.ready_for_editing == False:
			utils.customPrint(" ❗ Not ready for editing. Returning empty.")
			return []

		return self.foam_vectors

	def getFoamProperties(self):
		"""
		Returns the FoamProperties.
		"""

		if self.ready_for_editing == False:
			utils.customPrint(" ❗ Not ready for editing. Returning empty.")
			return {}

		return self.foam_properties

	def getFoamConfigurations(self):
		"""
		Returns the FoamConfigurations.
		"""

		if self.ready_for_editing == False:
			utils.customPrint(" ❗ Not ready for editing. Returning empty.")
			return {}

		return self.foam_configurations

	def _prepareFoamMesh(self):
		"""
		Prepare the FoamMesh for the OpenFOAM mesh.
		"""

		# Read the FoamMesh
		foam_mesh = self.foam_reader.readMeshToFoamMesh()
		foam_mesh._domain_type = self.parameters['domain type']

		return foam_mesh

	def _reconstructSimulationVariablesToSingleFiles(self, time_step_name):
		"""
		Reconstruct simulation variables to single files.
		"""

		# Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

		if utils_fenics_mpi.runningInSerialOrFirstProcessor():

			with utils_fenics_mpi.first_processor_lock():

				# If OpenFOAM ran in parallel, we need to reconstruct the
				 # full variables from the subdomains
				if self.RUN_IN_PARALLEL == True:
					if time_step_name == 'last':
						utils.run_command_in_shell("reconstructPar -case '%s' -latestTime" %(self.parameters['problem_folder']), mode = 'print directly to terminal', indent_print = True)
					else:
						utils.run_command_in_shell("reconstructPar -case '%s' -time %s" %(self.parameters['problem_folder'], time_step_name), mode = 'print directly to terminal', indent_print = True)

		# Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

	def _prepareFoamVectors(self, time_step_name = '0'):
		"""
		Prepare the FoamVectors for the OpenFOAM variables.
		"""

		utils.customPrint("\n 🌊 Preparing the FoamVectors...")

		################ If OpenFOAM ran in parallel ####################

		# If OpenFOAM ran in parallel, we need to reconstruct the
		 # full variables from the subdomains
		self._reconstructSimulationVariablesToSingleFiles(time_step_name)

		################ Variable names from solver ####################

		if self.location_solver != "":

			# File with the variable definitions in the solver
			variable_definition_file = "%s/createFields.H" %(self.location_solver)

			# Recognized variable types
			variable_types = ['volScalarField', 'volVectorField']

			# Get all names of the MUST_READ variables of the above types
			variable_names_from_solver = utils.getMustReadVariableNamesFromSolverFile(variable_definition_file, variable_types)

		################ Variable names from problem ###################

		variable_names_from_problem = utils.getNamesOfFilesInFolder("%s/0" %(self.parameters['problem_folder']), file_extensions_to_ignore = ['.py', '.swp'])
			# .py file  = Python version read from the OpenFOAM variable.
			# .swp file = Temporary file created when one file is left open. 

		###################### Check consistency #######################

		if utils.checkListInsideList(variable_names_from_solver, variable_names_from_problem) == False:
			raise ValueError(" ❌ ERROR: variable_names_from_problem == %s does not specify all variables required by variable_names_from_solver == %s" %(variable_names_from_problem, variable_names_from_solver))
		else:
			for i in range(len(variable_names_from_problem)):
				if variable_names_from_problem[i] in variable_names_from_solver:
					pass
				else:
					utils.customPrint(" ❗ Variable '%s' is not a \"MUST_READ\" variable." %(variable_names_from_problem[i]))
					break

		################## Create the FoamVectors ######################

		foam_vectors = [self.foam_reader.readVariableToFoamVector(variable_name, time_step_name = time_step_name) for variable_name in variable_names_from_problem]

		return foam_vectors

	def _prepareFoamProperties(self):
		"""
		Prepare the FoamProperties for the OpenFOAM properties.
		* Remember that all property files MUST end with 'Properties'.
		"""

		utils.customPrint("\n 🌊 Preparing the FoamProperties...")

		# Defined properties
		property_names = utils.getNamesOfFilesInFolder("%s/constant" %(self.parameters['problem_folder']), file_extensions_to_consider = ['Properties']) # file_extensions_to_ignore = ['.py'])

		# Create the FoamProperties
		foam_properties = [self.foam_reader.readPropertyToFoamProperty(property_name) for property_name in property_names]

		return foam_properties

	def _prepareFoamConfigurations(self):
		"""
		Prepare the FoamConfigurations for the OpenFOAM configurations.
		* Reading the FoamConfigurations from file is not implemented.
		  Therefore, you have to set everything through a dictionary
		  (i.e., can't reuse the previous configurations).
		"""

		utils.customPrint("\n 🌊 Preparing the FoamConfigurations...")

		# Defined configurations
		configuration_names = utils.getNamesOfFilesInFolder("%s/system" %(self.parameters['problem_folder']))

		## Remove blockMeshDict from the prepared configurations
		#configuration_names.remove('blockMeshDict')

		# Create the FoamConfigurations
		foam_configurations = [self.foam_reader.readConfigurationToFoamConfiguration(configuration_name) for configuration_name in configuration_names]
			# * Remember that 'foam_reader.readConfigurationToFoamConfiguration' 
			 #  returns an empty configuration object, because reading the 
			 #  corresponding OpenFOAM file is not implemented (at least for now).

		return foam_configurations

	########################## prepareNewFoamVector ########################

	def prepareNewFoamVector(self, variable_name, time_step_name = '0', skip_if_exists = False):
		"""
		Prepare a new FoamVector from file and include it in self.foam_vectors.
		The default is for generating for time step '0'.
		"""
		foam_vector_names = [foam_vector.name for foam_vector in self.foam_vectors]

		if variable_name in foam_vector_names:
			if skip_if_exists == True:
				utils.customPrint(" ❗ Skipping FoamVector '%s', which is already defined!..." %(variable_name))
				return
			else:
				raise ValueError(" ❌ ERROR: FoamVector '%s' is already defined!" %(variable_name))

		self.foam_vectors += [self.foam_reader.readVariableToFoamVector(variable_name, time_step_name = time_step_name)]

	####################### createFoamVectorFromFile #######################

	def createFoamVectorFromFile(self, variable_name, time_step_name = 'last'):
		"""
		Create a FoamVector from file, but not include it in self.foam_vectors.
		The default is for generating for time step 'last'.
		"""

		utils.customPrint("\n 🌊 Preparing FoamVector for '%s' (time step '%s')..." %(variable_name, time_step_name))
		return self.foam_reader.readVariableToFoamVector(variable_name, time_step_name = time_step_name)

	####################### _apply_changes_to_files ########################

	def _apply_changes_to_files(self):
		"""
		Apply changes to files.
		"""

		#### FoamVectors
		for foam_vector in self.foam_vectors:
			if foam_vector.check_if_to_apply_changes() == True:

				utils.customPrint("\n 🌊 Applying changes in FoamVector '%s'..." %(foam_vector.name))

				# Write FoamVector to file
				self.foam_writer.writeFoamVector(foam_vector)

				# Set that the changes have been applied!
				foam_vector.set_no_changes_to_apply()

		#### FoamProperties
		for foam_property in self.foam_properties:
			if foam_property.check_if_to_apply_changes() == True:

				utils.customPrint("\n 🌊 Applying changes in FoamProperty '%s'..." %(foam_property.name))

				# Write FoamProperty to file
				self.foam_writer.writeFoamProperty(foam_property)

				# Set that the changes have been applied!
				foam_property.set_no_changes_to_apply()

		#### FoamConfigurations
		for foam_configuration in self.foam_configurations:
			if foam_configuration.check_if_to_apply_changes() == True:
				
				utils.customPrint("\n 🌊 Applying changes in FoamConfiguration '%s'..." %(foam_configuration.name))

				# Write FoamConfiguration to file
				self.foam_writer.writeFoamConfiguration(foam_configuration)

				# Set that the changes have been applied!
				foam_configuration.set_no_changes_to_apply()

		#### FoamMesh
		if self.foam_mesh.check_if_to_apply_changes() == True:

			#### It is not recommended to apply changes to the mesh in the OpenFOAM format, 
			#    because one would have to make sure no inconsistency appears. Also, 
			#    since the FoamVectors depend on the mesh coordinates,
			#    they have to be 'manually' updated.

			utils.customPrint("\n 🌊 Applying changes in FoamMesh '%s'..." %(self.foam_mesh.name))

			# Write FoamConfiguration to file
			mesh_writer = FoamMeshWriter(self.foam_mesh)
			mesh_writer.write_to_foam()

			# Set that the changes have been applied!
			self.foam_mesh.set_no_changes_to_apply()

	############################## exportMesh ##############################

	def exportMesh(self, file_type = 'Ansys Fluent'):
		"""
		Export the mesh to a file.
		"""

		exportMesh(self.parameters['problem_folder'], file_type = file_type)

	###################### removeTimeStepFolders ###########################

	def removeTimeStepFolders(self, base_folder, keep_first_time_step = True):
		"""
		Remove time step folders.
		"""

		# Sorted OpenFOAM variable folders
		sorted_number_folders = utils.findFoamVariableFolders(base_folder)

		# The first folder name of 'sorted_number_folders' corresponds to the first time step
		if keep_first_time_step == True:
			sorted_number_folders = sorted_number_folders[1:]
		else:
			pass

		# Remove all remaining time step folders
		for time_step_name in sorted_number_folders:
			folder_to_remove = "%s/%s" %(base_folder, time_step_name)

			utils.customPrint(" 🌀 Removing folder '%s'" %(folder_to_remove))
			utils.removeFolderIfItExists(folder_to_remove)

	######################## computeTestUtility ############################

	def computeTestUtility(self, field_name = 'wallDist'):
		"""
		Compute a field by using a post-processing utility aimed for testing.

		-> Check out $FOAM_APP/test.

		-> The post-processing utilities computed by this function can be plotted
		   by running 'foam_solver.plotResults(...)'.

		-> Some of the OpenFOAM test utilities:
			'wallDist'	= [volScalarField] Compute the wall distance.
			 ...

		-> Also check 'foam_solver.computeField(...)' for the some more OpenFOAM fields.

		"""

		# [Parallel] Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

		utils.customPrint("\n 🌊 Checking the location of the OpenFOAM test utility '%s'..." %(field_name))

		# Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

		# Get the $FOAM_APP folder
		if utils_fenics_mpi.runningInSerialOrFirstProcessor():
			with utils_fenics_mpi.first_processor_lock():
				foam_applications_folder = utils.run_command_in_shell("echo -n $FOAM_APP", mode = 'save all prints to variable', indent_print = True)
		else:
			foam_applications_folder = None

		# Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

		# [Parallel] Broadcast
		if utils_fenics_mpi.runningInParallel():
			foam_applications_folder = utils_fenics_mpi.broadcastToAll(foam_applications_folder)

		# Module path
		module_path = "%s/test/%s" %(foam_applications_folder, field_name)

		# Check if test field is available.
		if utils.checkIfFileExists(module_path) == False:
			raise ValueError(" ❌ ERROR: OpenFOAM test utility '%s' does not exist in '%s'!" %(field_name, module_path))

		# Compile if needed
		 # * This is because OpenFOAM test utilities are not compiled by default. If you want to use any of them, 
		 #   you need to manually compile them (through wmake).
		if self.parameters['compile_modules_if_needed'] == True:
			module_name = utils.compileModuleIfNeeded(module_path, force_recompile = False, is_test_module = True)

		# Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

		# Compute the field
		utils.customPrint("\n 🌊 Computing OpenFOAM test utility '%s'..." %(field_name))
		if utils_fenics_mpi.runningInSerialOrFirstProcessor():
			with utils_fenics_mpi.first_processor_lock():
				utils.run_command_in_shell("%s -case '%s'" %(module_name, self.parameters['problem_folder']), mode = 'print directly to terminal', indent_print = True)

		# Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

	########################## computeCustomUtility ########################

	def computeCustomUtility(self, custom_utility_name = 'Urel', time_step_name = 'last'):
		"""
		Compute a custom utility from "fenics_topopt_foam/cpp_foam_utilities" folder.
		"""

		# Check if you want the last timestep
		if time_step_name == 'last':
			sorted_number_folders = utils.findFoamVariableFolders(self.parameters['problem_folder'])
			time_step_name = sorted_number_folders[len(sorted_number_folders) - 1]

		# Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

		if utils_fenics_mpi.runningInSerialOrFirstProcessor():
			with utils_fenics_mpi.first_processor_lock():
				module_paths_in_folder = ["%s/cpp_openfoam_utilities/%s" %(__fenics_topopt_foam_folder__, name) for name in os.listdir("%s/cpp_openfoam_utilities" %(__fenics_topopt_foam_folder__)) if os.path.isdir("%s/cpp_openfoam_utilities/%s" %(__fenics_topopt_foam_folder__, name))]

				assert "%s/cpp_openfoam_utilities/%s" %(__fenics_topopt_foam_folder__, custom_utility_name) in module_paths_in_folder, " ❌ ERROR: Utility '%s' is not available in module_paths_in_folder == %s" %(custom_utility_name, module_paths_in_folder)
		else:
			module_paths_in_folder = None

		# Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

		# [Parallel] Broadcast
		if utils_fenics_mpi.runningInParallel():
			module_paths_in_folder = utils_fenics_mpi.broadcastToAll(module_paths_in_folder)

		# Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

		if utils_fenics_mpi.runningInSerialOrFirstProcessor():
			with utils_fenics_mpi.first_processor_lock():
				utils.customPrint("\n 🌊 Computing utility '%s' for OpenFOAM..." %(custom_utility_name))
				utils.run_command_in_shell("%s -case '%s' -time %s" %(custom_utility_name, self.parameters['problem_folder'], time_step_name), mode = 'print directly to terminal', indent_print = True)
				 # * For running for the "last" time, "-time %s" can also be changed to "-latestTime"

		# Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

	############################# computeField #############################

	def computeField(self, field_name = 'yPlus', time_step_name = 'last'):
		"""
		Compute a field by using a post-processing utility.

		-> The post-processing utilities computed by this function can be plotted
		   by running 'foam_solver.plotResults(...)'.

		-> Some of the OpenFOAM utilities:
			-> Check 'Field calculation' in https://cfd.direct/openfoam/user-guide/v7-post-processing-cli/
			---
			'yPlus' 		= [volScalarField with values ONLY on boundary data (walls)] Compute y⁺.
			'wallShearStress' 	= [volVectorField with values ONLY on boundary data (walls)] Compute the shear stress at wall patches.
			'turbulenceIntensity'	= [volScalarField] Compute the turbulence intensity field I. 
			'vorticity'	        = [volVectorField] Calculates the vorticity field, i.e. the curl of the velocity field. 
			---
			'CourantNo'	        = [volScalarField] Calculates the Courant Number field from the flux field. 
			'MachNo'	        = [volScalarField] Calculates the Mach Number field from the velocity field. 
			'PecletNo'	        = [volScalarField] Calculates the Peclet Number field from the flux field. 
			---
			...

		-> Also check 'foam_solver.computeTestUtility(...)' for the OpenFOAM test fields.

		"""

		# Check if you want the last timestep
		if time_step_name == 'last':
			sorted_number_folders = utils.findFoamVariableFolders(self.parameters['problem_folder'])
			time_step_name = sorted_number_folders[len(sorted_number_folders) - 1]

		# Just in case, to give a hint of what to do.
		if field_name == 'residuals':
			utils.customPrint(" ❗ The utility 'residuals' does not seem to return anything (just N/A)... Use 'foam_solver.setToSaveResidualsToFile' before running the simulation, or use 'foam_solver.getResidualsFromLog' to get the residuals from an OpenFOAM log file.")
				# https://bugs.openfoam.org/view.php?id=2608
				# https://www.cfd-online.com/Forums/openfoam-post-processing/176644-plotting-residuals.html
				# https://www.cfd-online.com/Forums/openfoam-post-processing/191776-create-residual-field-postprocess.html

		# Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

		# Run '-postProcess' command from OpenFOAM
		if utils_fenics_mpi.runningInSerialOrFirstProcessor():
			with utils_fenics_mpi.first_processor_lock():
				utils.customPrint("\n 🌊 Computing OpenFOAM utility '%s'..." %(field_name))
				utils.run_command_in_shell("%s -postProcess -case '%s' -func %s -time %s" %(self.solver_name, self.parameters['problem_folder'], field_name, time_step_name), mode = 'print directly to terminal', indent_print = True)
				 # * For running for the "last" time, "-time %s" can also be changed to "-latestTime"

		# Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

	############################# plotResults ##############################

	def plotResults(self, file_type = 'VTK', tag_folder_name = "", time_step_name = 'last', rename_postProcessing_folder_too = True, rename_log_file_too = True, more_options_for_export = ""):
		"""
		Plot the results to files.
		-> If you want any additional variable to be plotted,
		   run 'foam_solver.computeField(...)'/'foam_solver.computeTestUtility(...)' before, for the desired variables.
		"""

		################ If OpenFOAM ran in parallel ####################

		# If OpenFOAM ran in parallel, we need to reconstruct the
		 # full variables from the subdomains
		self._reconstructSimulationVariablesToSingleFiles(time_step_name)

		################################################################

		# Export the results
		exportResults(self.parameters['problem_folder'], file_type = file_type, time_step_name = time_step_name, more_options = more_options_for_export)

		if type(tag_folder_name).__name__ == 'NoneType' or tag_folder_name == "":
			pass
		else:
			if file_type == 'VTK':

				# Remove destination 'VTK%s' folder if it already exists
				utils.removeFolderIfItExists('%s/VTK%s' %(self.parameters['problem_folder'], tag_folder_name))

				# If the results folder 'VTK' exists
				if utils.checkIfFileExists('%s/VTK' %(self.parameters['problem_folder'])) == True:

					# Wait for everyone!
					if utils_fenics_mpi.runningInParallel():
						utils_fenics_mpi.waitProcessors()

					# Rename the results folder 'VTK' to the destination folder 'VTK%s'
					if utils_fenics_mpi.runningInSerialOrFirstProcessor():
						with utils_fenics_mpi.first_processor_lock():
							utils.run_command_in_shell("mv '%s/VTK' '%s/VTK%s'" %(self.parameters['problem_folder'], self.parameters['problem_folder'], tag_folder_name), mode = 'print directly to terminal')

					# Wait for everyone!
					if utils_fenics_mpi.runningInParallel():
						utils_fenics_mpi.waitProcessors()

			else:
				raise ValueError(" ❌ ERROR: file_type == '%s' is not defined here!" %(file_type))

			# Also rename the postProcessing folder
			if rename_postProcessing_folder_too == True:

				# Remove destination 'postProcessing' folder if it already exists
				utils.removeFolderIfItExists('%s/postProcessing%s' %(self.parameters['problem_folder'], tag_folder_name))

				# If the results folder 'postProcessing' exists
				if utils.checkIfFileExists('%s/postProcessing' %(self.parameters['problem_folder'])) == True:

					# Wait for everyone!
					if utils_fenics_mpi.runningInParallel():
						utils_fenics_mpi.waitProcessors()

					# Rename the 'postProcessing' folder to the destination  'postProcessing'folder
					if utils_fenics_mpi.runningInSerialOrFirstProcessor():
						with utils_fenics_mpi.first_processor_lock():
							utils.run_command_in_shell("mv '%s/postProcessing' '%s/postProcessing%s'" %(self.parameters['problem_folder'], self.parameters['problem_folder'], tag_folder_name), mode = 'print directly to terminal')

					# Wait for everyone!
					if utils_fenics_mpi.runningInParallel():
						utils_fenics_mpi.waitProcessors()

			# Also rename the log file
			log_file = '%s/foam_log' %(self.parameters['problem_folder'])
			if rename_log_file_too == True:
				utils.renameFolderIfItExists('%s/%s' %(self.parameters['problem_folder'], log_file), new_name_type = 'new location', new_folder_location = '%s/%s%s' %(self.parameters['problem_folder'], log_file, tag_folder_name))

	##################### setToSaveResidualsToFile #########################

	def setToSaveResidualsToFile(self, foam_variable_names_that_are_solved = 'all', function_data_to_set = {}):
		"""
		Sets to save residuals to a file in the 'postProcessing' folder. 

		---

		-> This is not really needed, since all residuals are printed
		   to the log file when running OpenFOAM, such as with 

			foam_solver.solve(run_mode = 'openfoam', save_log_file = True)
			foam_solver.getResidualsFromLog()

		* Some differences are that, for the log file approach, 
		  the precision of the printed numbers is somewhat lower, 
		  and the results are not gathered in a single file. Also,
		  the "foamMonitor" utility can not be used.

		---

		-> Plot residuals while running OpenFOAM (or after finishing to run OpenFOAM):
				https://cfd.direct/openfoam/user-guide/v7-graphs-monitoring/

		Open another terminal and run:

			$ foamMonitor -l [problem_folder]/postProcessing/residuals/0/residuals.dat 

		---

		-> Check an example of how it is set in:

			$ foamInfo residuals
				-> option 2

		---

		https://cfd.direct/openfoam/user-guide/v6-graphs-monitoring/
		https://bugs.openfoam.org/view.php?id=2608
		https://www.cfd-online.com/Forums/openfoam-post-processing/176644-plotting-residuals.html
		https://www.cfd-online.com/Forums/openfoam-post-processing/191776-create-residual-field-postprocess.html

		"""

		# Find the controlDict
		for foam_configuration in self.foam_configurations:
			if foam_configuration.name == 'controlDict':
				foam_configuration_controlDict = foam_configuration

		# Use all variables that are defined by the user
		if foam_variable_names_that_are_solved == 'all':
			foam_variable_names_that_are_solved = [foam_vector.name for foam_vector in self.foam_vectors]

		# Prepare the fields that are saved to file
		fields = "("
		for i in range(len(foam_variable_names_that_are_solved)):
			variable_name = foam_variable_names_that_are_solved[i]
			fields += "%s" %(variable_name)
			if i < len(foam_variable_names_that_are_solved):
				fields += " "
		fields += ")"

		# Set the function name and data
		function_name = 'residuals'
		function_data = {
			'type' : 		'residuals',
			'functionObjectLibs' : "(\"libutilityFunctionObjects.so\")",
			'enabled': 		'true',
			'outputControl' : 	'timeStep',
			'outputInterval' : 	1,
			'fields' : 		fields,
		}
		function_data.update(function_data_to_set)

		# Set to configuration
		foam_configuration_controlDict.setFunctionToConfiguration(function_name, function_data)

		# Set to apply changes
		foam_configuration_controlDict.set_to_apply_changes('insert')

	####################### saveResidualsFromLog ###########################

	def saveResidualsFromLog(self, log_file = '', action_for_previous_log_folder = 'remove', silent_print = False, mpi_single_processor = False):
		"""
		Save the residuals from an OpenFOAM log.

		-> Since we will then have all variables in separated files,
		   we can open the corresponding file, "Select all", and "CTRL+C". Then, open a 
		   spreadsheet (such as LibreOffice Calc), and CTRL+V. Then, we will
		   have all the data in a column and can plot. Another alternative is 
		   using Matplotlib (by reading from the file, passing to a list and plotting).

		-> Updating the plots "on-the-fly" (i.e., while OpenFOAM is running):
			---
			1) GnuPlot
				https://www.cfdsupport.com/OpenFOAM-Training-by-CFD-Support/node88.html
				https://www.cfd-online.com/Forums/openfoam-community-contributions/64146-tutorial-how-plot-residuals.html
				https://www.cfd-online.com/Forums/openfoam-solving/111947-plot-residuals-fly.html
				http://lordvon64.blogspot.com/2014/10/how-to-plot-residuals-in-openfoam.html
				https://www.youtube.com/watch?v=ngyBbYn5-V0
			---
			2) PyFoam (pyFoamPlotWatcher.py)
				https://www.cfd-online.com/Forums/openfoam-post-processing/176644-plotting-residuals.html
			---
			3*) If you want to use the built-in OpenFOAM utility 'foamMonitor', check the function
		 	    'foam_solver.setToSaveResidualsToFile'.
			---
		https://www.researchgate.net/post/How_can_I_plot_the_residuals_graphically_in_openfoam

		-> Observation pertaining "cumulative error" value in OpenFOAM:
			-> cumulative error = the sum of all errors over previous iterations
				https://www.cfd-online.com/Forums/openfoam-solving/57839-simplefoam-time-step-continuity-errors-2.html

		"""

		if log_file == '':
			log_file = 'foam_log'
			# * Remember that 'foam_solver.plotResults' can rename the log file.

		log_file_full = '%s/%s' %(self.parameters['problem_folder'], log_file)

		log_folder = '%s/logs' %(self.parameters['problem_folder'])
		if utils.checkIfFileExists(log_folder, mpi_single_processor = mpi_single_processor) == True:
			if action_for_previous_log_folder == 'remove':
				utils.removeFolderIfItExists(log_folder, mpi_single_processor = mpi_single_processor)
			elif action_for_previous_log_folder == 'remove contents':
				utils.removeFolderContentIfItExists(log_folder, mpi_single_processor = mpi_single_processor)
			elif action_for_previous_log_folder == 'overwrite':
				pass
			else:
				raise ValueError(" ❌ ERROR: action_for_previous_log_folder == '%s' is not defined!" %(action_for_previous_log_folder))

		# Wait for everyone!
		if utils_fenics_mpi.runningInParallel(mpi_single_processor = mpi_single_processor):
			utils_fenics_mpi.waitProcessors()

		if silent_print == False:
			if utils_fenics_mpi.runningInSerialOrFirstProcessor():
				with utils_fenics_mpi.first_processor_lock(mpi_single_processor = mpi_single_processor):
					utils.customPrint("\n 🌊 Saving residuals from OpenFOAM log (%s)..." %(log_file_full))
					utils.run_command_in_shell("cd %s; foamLog '%s'" %(self.parameters['problem_folder'], log_file), mode = 'print directly to terminal', indent_print = True)

		else:
			if utils_fenics_mpi.runningInSerialOrFirstProcessor():
				with utils_fenics_mpi.first_processor_lock(mpi_single_processor = mpi_single_processor):
					utils.run_command_in_shell("cd %s; foamLog '%s'" %(self.parameters['problem_folder'], log_file), mode = 'no prints', indent_print = True)

		# Wait for everyone!
		if utils_fenics_mpi.runningInParallel(mpi_single_processor = mpi_single_processor):
			utils_fenics_mpi.waitProcessors()

	def plotAllResidualsFromLog(self, 
		x_axis_label = 'Time', 
		skip_first_residual_if_it_equals_to_1 = True,
		first_iterations_to_skip = 0,
		log_file = '', 
		action_for_previous_log_folder = 'remove',
		y_axis_scale = 'linear',
		silent_print = False,
		mpi_single_processor = False
		):
		"""
		Plot all available residuals from log folder.
		"""

		log_folder = '%s/logs' %(self.parameters['problem_folder'])
		if utils.checkIfFileExists(log_folder, mpi_single_processor = mpi_single_processor) == False:
			self.saveResidualsFromLog(log_file = log_file, action_for_previous_log_folder = action_for_previous_log_folder, silent_print = silent_print, mpi_single_processor = mpi_single_processor)

		# Names of all residual files
		names_of_the_residual_files = utils.getNamesOfFilesInFolder(log_folder, file_extensions_to_ignore = ['.awk'], mpi_single_processor = mpi_single_processor)

		# Plot all residual files
		for name_of_the_residual_file in names_of_the_residual_files:
			try:
				self.plotResidualsFromLog(
					name_of_the_residual_file, x_axis_label = x_axis_label, 
					skip_first_residual_if_it_equals_to_1 = skip_first_residual_if_it_equals_to_1,
					first_iterations_to_skip = first_iterations_to_skip,
					log_file = log_file,
					y_axis_scale = y_axis_scale,
					action_for_previous_log_folder = action_for_previous_log_folder,
					silent_print = silent_print,
					mpi_single_processor = mpi_single_processor
					)
			except:
				if silent_print == False:
					utils.customPrint("\n ❗ Unable to plot the contents of '%s'!" %(name_of_the_residual_file))
					import traceback
					traceback.print_exc()

	def plotResidualsFromLog(self, 
		name_of_the_residual_file, x_axis_label = 'Time', 
		skip_first_residual_if_it_equals_to_1 = True, 
		first_iterations_to_skip = 0, 
		log_file = '', 
		y_axis_scale = 'linear',
		action_for_previous_log_folder = 'remove',
		silent_print = False,
		mpi_single_processor = False
		):
		"""
		Plot residuals from log folder.
		"""

		# Save the residuals from log if not already saved
		log_folder = '%s/logs' %(self.parameters['problem_folder'])
		if utils.checkIfFileExists(log_folder, mpi_single_processor = mpi_single_processor) == False:
			self.saveResidualsFromLog(log_file = log_file, action_for_previous_log_folder = action_for_previous_log_folder, silent_print = silent_print, mpi_single_processor = mpi_single_processor)

		residual_file = '%s/%s' %(log_folder, name_of_the_residual_file)

		plot_object = None
		if utils.checkIfFileExists(residual_file, mpi_single_processor = mpi_single_processor) == True:

			if utils_fenics_mpi.runningInSerialOrFirstProcessor():
				with utils_fenics_mpi.first_processor_lock(mpi_single_processor = mpi_single_processor):

					if silent_print == False:
						utils.customPrint("\n 🌊 Plotting residuals from '%s'..." %(residual_file))

					assert 'plt' in globals() # Check if Matplotlib has been imported in the beginning of this file.

					# Load from file
					 # https://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html
					try:
						(x, y) = np.loadtxt(residual_file, delimiter='\t', usecols = (0, 1), unpack = True)
					except:
						if silent_print == False:
							utils.customPrint(" ❗ File '%s' is not in a 2-column format readable by NumPy. Can't plot" %(residual_file))
						return

					# Adjustment
					if 'float' in type(x).__name__ or 'int' in type(x).__name__:
						if silent_print == False:
							utils.customPrint(" ❗ File '%s' probably has a single value. Not plotting." %(residual_file))
						return

					if len(x) == 1:
						pass
					elif first_iterations_to_skip == 0:
						if skip_first_residual_if_it_equals_to_1 == True:
							if y[0] == 1.0:
								x = x[1:]
								y = y[1:]

					elif first_iterations_to_skip == 'first half':

						first_iterations_to_skip = len(x)//2
							# * "//" = Integer division in Python

						x = x[first_iterations_to_skip:]
						y = y[first_iterations_to_skip:]

					else:
						assert 'int' in type(first_iterations_to_skip).__name__
						x = x[first_iterations_to_skip:]
						y = y[first_iterations_to_skip:]

					# Create plot
					plot_object = plt.plot(x,y)

					# Title
					plt.title('%s x %s' %(name_of_the_residual_file, x_axis_label))

					# Labels
					plt.xlabel('%s' %(x_axis_label))
					plt.ylabel('%s' %(name_of_the_residual_file))

					# Set scale
					if y_axis_scale == 'linear':
						pass
					elif y_axis_scale == 'log/symlog':
						if y.min() == 0:
							utils.customPrint("❗Log/symlog scaling impossible, because there is a zero-valued number!")
						elif y.min() < 0:
							min_delta = np.abs(y).min()
							plt.yscale('symlog', linthreshy = min_delta)
								# MatplotlibDeprecationWarning: The 'linthreshy' parameter of __init__() has been renamed 'linthresh' since Matplotlib 3.3; support for the old name will be dropped two minor releases later.
								# Ok 👍️. Anyway, it doesn't hurt to keep this warning for now, because I am still running Matplotlib < 3.3 in other installations...
						else:
							plt.yscale('log')
					else:
						raise ValueError(" ❌ ERROR: y_axis_scale == '%s' is not defined here!" %(y_axis_scale))

					# Adjust everything inside the plot area
					plt.tight_layout()

					# Export to .png
					plt.savefig('%s_plot.png'  % (residual_file))

					# Close the plot
					plt.close()

		else:
			if silent_print == False:
				utils.customPrint("❗File '%s' does not exist!" %(residual_file))

		return plot_object

	######################### setToRunInParallel ###########################

	def setToRunInParallel(self, parallel_data, set_foam_file_info = True, set_new_data = True, variable_environment_reinitialization_for_parallel = None):
		"""
		Sets up OpenFOAM for running in parallel.

		https://cfd.direct/openfoam/user-guide/v7-running-applications-parallel/
		https://www.openfoam.com/documentation/user-guide/running-applications-parallel.php

		https://github.com/openfoamtutorials/openfoam_tutorials/tree/master/Parallel
		https://github.com/openfoamtutorials/openfoam_tutorials/blob/master/Parallel/case/system/decomposeParDict
		https://github.com/openfoamtutorials/openfoam_tutorials/blob/master/Parallel/decomposeParDict
		"""

		utils.customPrint("\n 🌊 Setting OpenFOAM to run in parallel...")

		# Create a writer for the OpenFOAM files
		if 'foam_writer' not in self.__dict__:
			self.foam_writer = FoamWriter(self.parameters['problem_folder'], python_write_precision = self.python_write_precision)

		if set_new_data == True:

			# Configuration file name
			configuration_file_name = 'decomposeParDict'

			# FoamFile
			data_to_add = {}
			if set_foam_file_info == True:

				data_to_add['FoamFile'] = {

					'version' : '2.0',
					'format' : 'ascii',
					'class' : 'dictionary',
					'location' : '"system"',
					'object' : configuration_file_name,

				}

				for key in parallel_data:
					data_to_add[key] = parallel_data[key]

			else:

				for key in parallel_data:
					data_to_add[key] = parallel_data[key]

				assert 'FoamFile' in data_to_add

			# Number of subdomains
			assert 'numberOfSubdomains' in data_to_add

			# Write the data to configuration file
			self.foam_writer.writeDataToFile(
				data_to_add = data_to_add, 
				file_to_use = "%s/system/%s" % (self.parameters['problem_folder'], configuration_file_name)
			)

			# Set the new configuration
			configuration_exists = False
			for foam_configuration in self.foam_configurations:
				if foam_configuration.name == 'decomposeParDict':
					configuration_exists = True
					current_foam_configuration = foam_configuration
					break

			if configuration_exists == True:
				current_foam_configuration.reloadData(data_to_add, configuration_file_name, set_foam_file_info = False)
			else:
				new_foam_configuration = FoamConfiguration(data_to_add)
				self.foam_configurations += [new_foam_configuration]

		#### Set to run in parallel
		self.RUN_IN_PARALLEL = True

		#### Shell script lines to set OpenFOAM solve command in parallel

		# Reinitialization of the environment variables, 
		 # because we need to "clean" MPI-related 
		 # variables in order to be able to run a separated MPI process 
		 # from the current one set by FEniCS.

		# Check if the user has provided the reinitialization of the Shell variable environment.
		 # If not, there is a default setup defined below.
		if type(variable_environment_reinitialization_for_parallel).__name__ == 'NoneType':
			variable_environment_reinitialization_for_parallel = {
				'environment scripts' : ["$F_DIR_FENICS/f.sh", "$FOAM_ETC/bashrc"],
					# $F_DIR_FENICS/f.sh is just a script that I use to initialize the FEniCS environment, If you don't have it, the code will simply ignore it!
					# $FOAM_ETC/bashrc is always needed, given that you are using OpenFOAM
				'additional setup inside environment' : "", #"export -f foamVersion",
			}
		self.variable_environment_reinitialization_for_parallel = variable_environment_reinitialization_for_parallel

		# The number of subdomains is the number of processes that will be used by MPI for OpenFOAM
		self.NUMBER_OF_SUBDOMAINS = parallel_data['numberOfSubdomains']

	####################### unsetFromRunInParallel #########################

	def unsetFromRunInParallel(self):
		"""
		Set OpenFOAM not to run in parallel.
		* Just in case you got tired of running OpenFOAM in parallel. =)
		"""

		self.RUN_IN_PARALLEL = False
		self.variable_environment_reinitialization_for_parallel = {}
		self.NUMBER_OF_SUBDOMAINS = None

	############################### solve ##################################

	def solve(self, 
		run_mode = 'openfoam', 
		force_rewrite_if_necessary = False, 
		consider_that_we_may_have_successive_simulation_parameters = True, 

		# Log file
		save_log_file = True, 
		log_file = '',
		only_print_to_log_file = False,

		# Silent mode
		silent_run_mode = False,
		num_logfile_lines_to_print_in_silent_mode = 0,

		# MPI configs
		mpi_configs = {},

		# Continuous plotting of residuals
		continuously_plot_residuals_from_log = False,
		continuously_plot_residuals_from_log_tag = '',
		continuously_plot_residuals_from_log_time_interval = 5,
		continuously_plot_residuals_from_log_x_axis_label = 'Time',
		continuously_plot_residuals_from_log_y_axis_scale = 'linear',
		continuously_plot_residuals_from_log_use_lowest_priority_for_plotting = True,
		):
		"""
		Solve the problem with OpenFOAM.
		If you want to run OpenFOAM in parallel, run the function
		'foam_solver.setToRunInParallel(...)' before this function.
		"""

		additional_kwargs = {}

		# Set up continuous plotting of residuals
		if continuously_plot_residuals_from_log == True:

			if log_file == '':
				log_file = 'foam_log'

			log_file = '%s%s' %(log_file, continuously_plot_residuals_from_log_tag)
			log_file_path = '%s/%s' %(self.parameters['problem_folder'], log_file)

			def auxiliary_function_in_fork():
				"""
				Plot residuals from log file.
				"""

				if os.path.exists(log_file_path):

					# Save residuals from foam_log
					self.saveResidualsFromLog(log_file = log_file, action_for_previous_log_folder = 'overwrite', silent_print = True, mpi_single_processor = True)

					# Plota all residuals from foam_log
					self.plotAllResidualsFromLog(
						x_axis_label = continuously_plot_residuals_from_log_x_axis_label, 
						skip_first_residual_if_it_equals_to_1 = True, 
						first_iterations_to_skip = 0, 
						log_file = log_file,
						action_for_previous_log_folder = 'overwrite',
						y_axis_scale = continuously_plot_residuals_from_log_y_axis_scale,
						silent_print = True,
						mpi_single_processor = True
						)

				return True # Continue looping

			additional_kwargs['auxiliary_function_in_fork'] = auxiliary_function_in_fork
			additional_kwargs['time_interval_auxiliary_function_in_fork'] = continuously_plot_residuals_from_log_time_interval
			additional_kwargs['use_lowest_priority_in_fork'] = continuously_plot_residuals_from_log_use_lowest_priority_for_plotting

		if consider_that_we_may_have_successive_simulation_parameters == False:

			# Solve with FoamSolver
			result_info = self.__solve(run_mode = run_mode, save_log_file = save_log_file, force_rewrite_if_necessary = force_rewrite_if_necessary, log_file = log_file, only_print_to_log_file = only_print_to_log_file, silent_run_mode = silent_run_mode, num_logfile_lines_to_print_in_silent_mode = num_logfile_lines_to_print_in_silent_mode, mpi_configs = mpi_configs, **additional_kwargs)

		else:

			def recursiveFindSuccessiveSimulationParameters(dictionary, current_location = ''):
				"""
				Recusively find all SuccessiveSimulationParameters.
				(* And also set tags to print the location of the SuccessiveSimulationParameters...)
				"""
				found = []
				for key in dictionary:
					if type(dictionary[key]).__name__ == 'dict':
						found += recursiveFindSuccessiveSimulationParameters(dictionary[key], current_location = "%s » '%s'" %(current_location, key)) # Recursion
					elif type(dictionary[key]).__name__ == 'SuccessiveSimulationParameter':
						if dictionary[key].size >= 1:
							found += [dictionary[key]]
							found[len(found) - 1].set_tag("%s » '%s'" %(current_location, key))
						else:
							pass # We don't need to update a single-valued 'SuccessiveSimulationParameter' instance
					else:
						pass
				return found

			found_successive = []
			configs_to_set_to_update = []

			# Browse all properties and configurations for SuccessiveSimulationParameters
			foam_configurations = self.getFoamConfigurations()
			for foam_configuration in foam_configurations:
				found = recursiveFindSuccessiveSimulationParameters(foam_configuration.data, current_location = "'%s'" %(foam_configuration.name))
				if len(found) > 0:
					found_successive += found
					configs_to_set_to_update += [foam_configuration]

			foam_properties = self.getFoamProperties()
			for foam_property in foam_properties:
				found = recursiveFindSuccessiveSimulationParameters(foam_property.data, current_location = "'%s'" %(foam_property.name))
				if len(found) > 0:
					found_successive += found
					configs_to_set_to_update += [foam_property]

			if consider_that_we_may_have_successive_simulation_parameters == False or len(found_successive) == 0:

				# Solve with FoamSolver
				result_info = self.__solve(run_mode = run_mode, save_log_file = save_log_file, force_rewrite_if_necessary = force_rewrite_if_necessary, log_file = log_file, only_print_to_log_file = only_print_to_log_file, silent_run_mode = silent_run_mode, num_logfile_lines_to_print_in_silent_mode = num_logfile_lines_to_print_in_silent_mode, mpi_configs = mpi_configs, **additional_kwargs)

			else:

				# Let's overload the 'error_on_nonconvergence' parameter from FoamSolver
				error_on_nonconvergence_BAK = self.parameters['error_on_nonconvergence']
				self.parameters['error_on_nonconvergence'] = False

				# Set for the first value
				[found_successive[i].restart() for i in range(len(found_successive))]

				# Create a copy of the first folder
				sorted_number_folders = utils.findFoamVariableFolders(self.parameters['problem_folder'])
				first_time_step = sorted_number_folders[0]
				initial_file_path = '%s/%s' %(self.parameters['problem_folder'], first_time_step)
				bak_initial_file_path = '%s/bak_%s' %(self.parameters['problem_folder'], first_time_step)

				utils.removeFolderIfItExists(bak_initial_file_path)
				utils.copyFolder(initial_file_path, bak_initial_file_path, overwrite = False)

				file_names_in_folder = utils.getFileNamesInFolder(bak_initial_file_path)

				def checkIfAnySuccessiveSimulationParameterIsUpdatable(found_successive):
					return any([found_successive[i].checkIfUpdatable() for i in range(len(found_successive))])

				max_count_steps = max([found_successive[i].size for i in range(len(found_successive))])

				first_run_flag = True
				count_steps = 0
				while checkIfAnySuccessiveSimulationParameterIsUpdatable(found_successive) == True:

					# Increment step count
					count_steps += 1

					# Print current step
					utils.customPrint("""
 ══════════════════════════════════════════════════════════════════════════════
 🌊 Running successive simulation %d/%d ...

""" %(count_steps, max_count_steps))
					# Check (i.e. print) the next values that will be set
					[found_successive[i].checkNextValue() for i in range(len(found_successive))]

					if first_run_flag == True:
						first_run_flag = False
					else:

						#### If OpenFOAM ran in parallel
						 # * Is there a more computationally efficient way to do it? Yes, there is:
						 #   We would need to copy the folders that ran in parallel over
						 #   the initial folders that have already been split for parallelism,
						 #   and then set foam_solver not to recreate the split files.

						# If OpenFOAM ran in parallel, we need to reconstruct the
						 # full variables from the subdomains
						self._reconstructSimulationVariablesToSingleFiles('last')

						#### Set the last results to the first iteration folder

						# Last time step
						sorted_number_folders = utils.findFoamVariableFolders(self.parameters['problem_folder'])
						last_time_step = sorted_number_folders[len(sorted_number_folders) - 1]

						if last_time_step != first_time_step:

							# Remove the first folder
							utils.removeFolderIfItExists(initial_file_path)

							# Recreate the first folder
							utils.createFolderIfItDoesntExist(initial_file_path)

							# Copy the last time step to the first time step
							utils.copyFilesFromFolderToNewLocation(self.parameters['problem_folder'], destination_folder_name = first_time_step, original_folder_name = last_time_step, file_names_in_folder = file_names_in_folder)

						else:
							utils.customPrint(" ❗ No results generated! Reusing previous results as initial guess!")

					# Set to write
					[configs_to_set_to_update[i].set_to_apply_changes('insert') for i in range(len(configs_to_set_to_update))]

					# Solve with FoamSolver
					result_info = self.__solve(run_mode = run_mode, save_log_file = save_log_file, force_rewrite_if_necessary = True, log_file = log_file, only_print_to_log_file = only_print_to_log_file, silent_run_mode = silent_run_mode, num_logfile_lines_to_print_in_silent_mode = num_logfile_lines_to_print_in_silent_mode, mpi_configs = mpi_configs, **additional_kwargs)

					# Print current step
					utils.customPrint("""

 ══════════════════════════════════════════════════════════════════════════════
""")

				# Replace the first folder back
				utils.removeFolderIfItExists(initial_file_path)
				utils.copyFolder(bak_initial_file_path, initial_file_path, overwrite = False)

				# Set for the first value
				[found_successive[i].restart() for i in range(len(found_successive))]

				# Now let's return the previous 'error_on_nonconvergence' parameter from FoamSolver
				self.parameters['error_on_nonconvergence'] = error_on_nonconvergence_BAK

				# Check convergence
				converged = result_info[1]
				if converged == False:
					if self.parameters['error_on_nonconvergence'] == True:
						last_time_step = result_info[0]
						raise ValueError(" ❌ ERROR: OpenFOAM solver did not converge until time step %s!" %(last_time_step))
					else:
						if self.parameters['error_on_nonconvergence_failure'] == True:
							last_lines = utils.getLastLinesOfFile(log_file, number_of_lines = 5)
							if ("?" in last_lines) or ('FATAL' in last_lines):
								raise ValueError(" ❌ ERROR: OpenFOAM solver did not converge until time step %s, and even DIVERGED (or some error occurred)!" %(last_time_step))

		return result_info

	############################# __solve ##################################

	@utils.createForkForAuxiliaryFunction()
	def __solve(self, 
		run_mode = 'openfoam', 
		force_rewrite_if_necessary = False, 

		# Log file
		save_log_file = True, 
		log_file = '', 
		only_print_to_log_file = False, 

		# Silent mode
		silent_run_mode = False,
		num_logfile_lines_to_print_in_silent_mode = 0,

		# MPI configs
		mpi_configs = {},
		):
		"""
		Backend for solving the problem with OpenFOAM. Use foam_solver.solve .
		"""

		# Apply necessary edits
		if self.ready_for_editing == True or force_rewrite_if_necessary == True:
			self._apply_changes_to_files()
			self.ready_for_editing = False

		# Find the controlDict
		for foam_configuration in self.foam_configurations:
			if foam_configuration.name == 'controlDict':
				foam_configuration_controlDict = foam_configuration

		# Check if 'endTime' and 'writeInterval' are fine!
		if ('endTime' in foam_configuration_controlDict.data) and ('writeInterval' in foam_configuration_controlDict.data):

			stopAt_config = foam_configuration_controlDict.data['stopAt']
			stopAt = stopAt_config if type(stopAt_config).__name__ != 'SuccessiveSimulationParameter' else stopAt_config.current_value

			if stopAt_config == 'endTime':

				endTime_config = foam_configuration_controlDict.data['endTime']
				endTime = float(endTime_config) if type(endTime_config).__name__ != 'SuccessiveSimulationParameter' else float(endTime_config.current_value)

				writeInterval_config = foam_configuration_controlDict.data['writeInterval']
				writeInterval = float(writeInterval_config) if type(writeInterval_config).__name__ != 'SuccessiveSimulationParameter' else float(writeInterval_config.current_value)

				if endTime < writeInterval:
					raise ValueError(" ❌ ERROR: endTime '%s' is smaller than writeInterval '%s'. Therefore, the output will ALWAYS be nothing! (* Well, even if there are no results, you would still be able to see the prints of the residuals, but who would want just that...)" %(endTime, writeInterval))

		# Clean up in case you are reusing a previous folder structure
		self.removeTimeStepFolders(self.parameters['problem_folder'], keep_first_time_step = True)

		# Clean up log folder
		log_folder = '%s/logs' %(self.parameters['problem_folder'])
		if utils.checkIfFileExists(log_folder) == True:
			utils.removeFolderIfItExists(log_folder)

		# Clean up and setup in the case of running in parallel
		if self.RUN_IN_PARALLEL == True:

			# Process folders, which are generated for 
			 # running in parallel (such as 'processor0', 'processor1' etc.).
			subfolder_names = utils.getSubfolderNames(self.parameters['problem_folder'], startswith = 'processor')

			# Remove all of process folders
			if len(subfolder_names) == 0:
				pass
			else:
				for subfolder_name in subfolder_names:
					utils.removeFolderIfItExists("%s/%s" %(self.parameters['problem_folder'], subfolder_name))

				## Remove the time steps of all process folders, which are generated for 
				# # running in parallel (such as 'processor0', 'processor1' etc.).
				#for subfolder_name in subfolder_names:
				#	self.removeTimeStepFolders("%s/%s" %(self.parameters['problem_folder'], subfolder_name), keep_first_time_step = True)

			# Wait for everyone!
			if utils_fenics_mpi.runningInParallel():
				utils_fenics_mpi.waitProcessors()

			if utils_fenics_mpi.runningInSerialOrFirstProcessor():
				with utils_fenics_mpi.first_processor_lock():
					utils.customPrint("\n 🌊 Decomposing the domain for OpenFOAM to run in parallel...")
					utils.run_command_in_shell("decomposePar -case '%s'" %(self.parameters['problem_folder']), mode = 'print directly to terminal', indent_print = True)

			# Wait for everyone!
			if utils_fenics_mpi.runningInParallel():
				utils_fenics_mpi.waitProcessors()

		# Run OpenFOAM
		foam_run_command = '%s -case %s' %(self.solver_name, self.parameters['problem_folder'])

		# Set for running in parallel
		if self.RUN_IN_PARALLEL == True:

			# If you are using FEniCS TopOpt Foam in a cluster, you may end up in any
			# of the following two situations, in which different countermeasures may
			# be performed in order to run the code:
				#
				# 1) "The scratch folder is temporary ("created for the job") and can access the home folder 
				#     during the execution of the job":
				#     Write a script that loads all modules and everything else that is needed, 
				#     and set it to run by including it in the variable 
				#     'variable_environment_reinitialization_for_parallel' in foam_solver.setToRunInParallel
				#
				# 2) "The scratch folder is always the same and cannot access the home folder during 
				#    the execution of the job":
				#    Create a .bashrc file in the scratch folder, which loads all modules and everything 
				#    else that is needed. This way, it is not needed to use workarounds such as using
				#    'foamExec' or the full location of the executables
				#    (full location of the executables: $(type -p mpirun), $(type -p simpleFoam), $(type -p foamExec) etc.).
				#

			# Default is to simply use the number of subdomains as the number of processes
			if len(mpi_configs) == 0: 
				mpi_configs = {

					# Number of processes
					'-np' : 'NUMBER_OF_SUBDOMAINS',

					# Display allocation, because it makes it easier
					 # to spot mistakes made in the setup (in the case the OpenFOAM
					 # simulation fails due to an mpi related issue)
					'-v' : None, # Display verbose
					'--display-allocation ' : None, # Display allocation of nodes

				}

			# Please, don't include this "unknown" parameter here...
			assert ('-parallel' not in mpi_configs)

			mpirun_command = 'mpirun'

			for key in mpi_configs:
				if type(mpi_configs[key]).__name__ == 'NoneType':
					mpirun_command += ' %s' %(key)
				elif mpi_configs[key] == 'NUMBER_OF_SUBDOMAINS':
					mpirun_command += ' %s %d' %(key, self.NUMBER_OF_SUBDOMAINS)
				else:
					mpirun_command += ' %s %s' %(key, mpi_configs[key])

			foam_run_command = '%s %s -parallel' %(mpirun_command, foam_run_command)

		# Set according to the run_mode
		if run_mode == 'openfoam':

			utils.customPrint("\n 🌊 Solving the problem with OpenFOAM...")

		elif type(run_mode).__name__ == 'function':

			foam_run_command = run_mode(foam_run_command)

		else:
			raise ValueError(" ❌ ERROR: run_mode == '%s' is not defined!" %(run_mode))

		# Adjust to generate log file
		if save_log_file == True:

			if log_file == '':
				log_file = 'foam_log'

			log_file_path = '%s/%s' %(self.parameters['problem_folder'], log_file)

			if only_print_to_log_file == True:
				foam_run_command = '( %s ) 2>&1 > %s' %(foam_run_command, log_file_path)
			else:
				foam_run_command = '( %s ) 2>&1 | tee %s' %(foam_run_command, log_file_path)

		# Setup for the Shell variables when running in parallel
		if self.RUN_IN_PARALLEL == True:
			foam_run_command = utils.setToReinitializeShellEnvironment(
				foam_run_command, 
				variable_environment_reinitialization = self.variable_environment_reinitialization_for_parallel
				)

		# Init time count
		utils.initTimeCount('OpenFOAM simulation')

		# Silent mode
		if silent_run_mode == False:
			run_mode_in_shell = 'print directly to terminal'
		else:
			run_mode_in_shell = 'no prints'
			utils.customPrint(" 🌀 [Start: %s] Running OpenFOAM simulation in silent mode..." %(str(datetime.now())))

		# Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

		# Solve
		if utils_fenics_mpi.runningInSerialOrFirstProcessor():
			with utils_fenics_mpi.first_processor_lock():
				utils.run_command_in_shell(foam_run_command, mode = run_mode_in_shell, indent_print = True)

		# Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

		# Silent mode
		if silent_run_mode == False:
			pass
		else:
			if save_log_file == True and num_logfile_lines_to_print_in_silent_mode > 0:

				# Wait for everyone!
				if utils_fenics_mpi.runningInParallel():
					utils_fenics_mpi.waitProcessors()

				# Print the last lines
				if utils_fenics_mpi.runningInSerialOrFirstProcessor():
					with utils_fenics_mpi.first_processor_lock():
						utils.run_command_in_shell("tail --lines %d %s" %(num_logfile_lines_to_print_in_silent_mode, log_file_path), mode = 'print directly to terminal', indent_print = True, suppress_run_print = True)
						# /usr/bin/tail

				# Wait for everyone!
				if utils_fenics_mpi.runningInParallel():
					utils_fenics_mpi.waitProcessors()

			else:
				pass

			utils.customPrint(" 🌀 [Finish: %s] OpenFOAM simulation finished!" %(str(datetime.now())))

		# Finish time count
		utils.finishTimeCount('OpenFOAM simulation')

		############### Check if it has converged or not ###############

		# Last time step
		if self.RUN_IN_PARALLEL == True:
			sorted_number_folders = utils.findFoamVariableFolders("%s/processor0" %(self.parameters['problem_folder']))
			last_time_step = sorted_number_folders[len(sorted_number_folders) - 1]
		else:
			sorted_number_folders = utils.findFoamVariableFolders(self.parameters['problem_folder'])
			last_time_step = sorted_number_folders[len(sorted_number_folders) - 1]

		if save_log_file == True: # Check the log file for the word 'converged'

			# Results line
			results_line = utils.run_command_in_shell("echo -n $(grep \"converged\" %s)" %(log_file_path), mode = 'save output to variable', accept_empty_response = True)

			# Converged
			if results_line == "":
				converged = False
				if self.parameters['error_on_nonconvergence'] == True:
					raise ValueError(" ❌ ERROR: OpenFOAM solver did not converge until time step %s! If this is not the last iteration, you may want to set the \"writeInterval\" parameter of controlDict to a smaller value (such as 1) in order to be able to check the result of the last step." %(last_time_step))
				else:
					if self.parameters['error_on_nonconvergence_failure'] == True:
						last_lines = utils.getLastLinesOfFile(log_file_path, number_of_lines = 10)# = 5)
						if ("?" in last_lines) or ('FATAL' in last_lines):
							raise ValueError(" ❌ ERROR: OpenFOAM solver did not converge until time step %s, and even DIVERGED (or some error occurred)! If this is not the last iteration, you may want to set the \"writeInterval\" parameter of controlDict to a smaller value (such as 1) in order to be able to check the result of the last step." %(last_time_step))

					utils.customPrint(" ❌ OpenFOAM solver did not converge until time step %s! If this is not the last iteration, you may want to set the \"writeInterval\" parameter of controlDict to a smaller value (such as 1) in order to be able to check the result of the last step." %(last_time_step))
			else:
				converged = True
				utils.customPrint(" ✅ OpenFOAM solver converged in %s steps!" %(last_time_step))

			return [last_time_step, converged]
		else:
			utils.customPrint(" ❗ Can't check convergence without generating OpenFOAM log!")
			return [last_time_step, True] # Sorry, without the log file, we can not determine if it converged or not

############################## Load plugins ####################################

from ..plugins import load_plugins
load_plugins.loadPlugins(__file__, globals())

################################################################################

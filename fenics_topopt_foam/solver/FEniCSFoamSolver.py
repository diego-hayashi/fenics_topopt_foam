################################################################################
#                              FEniCSFoamSolver                                #
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

# FoamSolver
from .FoamSolver import FoamSolver

# Mesh
from ..mesh.createMeshFromFEniCS import createMeshFromFEniCS

# I/O (Input/Output)
from ..io.FoamWriter import FoamWriter
from ..io.FoamReader import FoamReader

# Utilities
from ..utils import utils

# FEniCS utilities
from ..utils import utils_fenics

# FEniCS+MPI utilities
from ..utils import utils_fenics_mpi

############################### FEniCS libraries ###############################

# FEniCS
try:
	from fenics import *
	 # Do not import dolfin-adjoint here. This is a non-annotated module!
except:
	utils.printDebug(" ❌ FEniCS not installed! Can't use FEniCSFoamSolver!")

############################# FEniCSFoamSolver #################################

class FEniCSFoamSolver():
	"""
	Solver structure for interfacing OpenFOAM with FEniCS.
	* All functions here should contain the decorator "@utils_fenics.no_annotations()"
	  to guarantee that, if we are using dolfin-adjoint, no annotations will
	  be performed. This decorator corresponds to the "no_annotations" decorator
	  of dolfin-adjoint when you have dolfin-adjoint installed; otherwise, it
	  does nothing.
	"""

	@utils_fenics.no_annotations()
	def __init__(self, 
		mesh, boundary_data, 			# FEniCS mesh and boundary data
		parameters, 				# FoamSolver parameters
		properties_dictionary,			# OpenFOAM properties set inside a dictionary
		configurations_dictionary,		# OpenFOAM configurations set inside a dictionary

		# Mesh
		use_mesh_from_foam_solver = False, 	# Choose this as True ONLY if you already have the mesh defined in the OpenFOAM format

		# Problem setup
		create_all_problem_setup = True,	# Choose this as False ONLY if you want to reuse files that are already in the OpenFOAM folder structure
		action_to_do_to_create_all_problem_setup = 'remove only the initial guess folder and overwrite whatever needed', # Choose what you want to do in case the problem folder already exists
			# 'overwrite whatever needed'
			# 'remove only the initial guess folder and overwrite whatever needed'
			# 'remove previous problem folder'
			# 'remove all previous files'
			# 'rename previous problem'

		# Mapping
		tol_factor_for_mapping_FEniCS_and_OpenFOAM = 0.005, # Tolerance factor for matching FEniCS and OpenFOAM

		# Projection setup
		projection_setup = {}, 

		# Write precision in the Python code
		python_write_precision = 6,

		# Configuration for measurement units
		configuration_of_openfoam_measurement_units = {
			'pressure' : 'rho-normalized pressure',
					# * It is assumed that FEniCS always uses "Pa".
					# * Remember: This configuration should match the OpenFOAM solver that you are using!
				# 'rho-normalized pressure' = Pressure unit is converted from "Pa" to "Pa/(kg/m³)" for OpenFOAM. Used in incompressible solvers, such as "simpleFoam", "SRFSimpleFoam" etc.
				# 'pressure' = Pressure unit is "Pa" for OpenFOAM. Used in compressible solvers, such as "rhoSimpleFoam", "rhoPimpleFoam" etc.
		},

		# Additional measurement units
		additional_measurement_units = {},

		# Minimum value for the projected turbulence variables
		min_value_projected_turbulence_variables = 1.E-14
		):

		utils.customPrint("\n 🌊 Creating FEniCSFoamSolver...", mpi_wait_for_everyone = True)

		# Dictionary of measurement units
		self.dictionary_measurement_units = {

			# Velocity
			'U' : 		'velocity',

			# Relative velocity
			'Urel' : 	'velocity',

			# Other velocities
			'U_*' : 	'velocity',

			# Pressure
			'p' : 		'rho-normalized pressure',

			# Temperature
			'T' : 		'temperature',

			# Kinematic viscosity
			'nu' : 		'kinematic viscosity', 

			# Design variable
			'alpha_design' : 'dimensionless',

			# Turbulent variables
			'k' : 		'specific turbulent kinetic energy', 					# k-epsilon, k-omega models
			'epsilon' : 	'specific rate of dissipation of turbulent kinetic energy', 		# k-epsilon model
			'omega' : 	'specific frequency of dissipation of turbulent kinetic energy', 	# k-omega model

			'nuTilda' : 	'kinematic viscosity', # Spalart-Allmaras model

			'v2' : 		'fluctuating velocity normal to the streamlines', 	# v2-f model
			'f' : 		'relaxation function', 					# v2-f model

			# Turbulent kinematic viscosity
			'nut' : 		'kinematic viscosity', 

			# Thermal diffusivity multiplied by the density
			'alphat' : 		'thermal diffusivity multiplied by the density',

			# Some more unit specifications
			'yWall_to_load' : 	'length',
			'nWall_to_load' : 	'dimensionless',

		}
		if configuration_of_openfoam_measurement_units['pressure'] == 'rho-normalized pressure':
			self.dictionary_measurement_units['p'] = 'rho-normalized pressure'
		elif configuration_of_openfoam_measurement_units['pressure'] == 'pressure':
			self.dictionary_measurement_units['p'] = 'pressure'
		else:
			raise ValueError(" ❌ ERROR: configuration_of_openfoam_measurement_units['compressibility'] == '%s' is not defined!" %(configuration_of_openfoam_measurement_units['compressibility']))

		if len(additional_measurement_units) > 0:
			self.dictionary_measurement_units.update(additional_measurement_units)

		# Minimum value for the projected turbulence variables
		self.min_value_projected_turbulence_variables = min_value_projected_turbulence_variables

		# Problem folder
		problem_folder = parameters['problem_folder']
		self.problem_folder = problem_folder

		# Tolerance factor for matching FEniCS and OpenFOAM
		self.tol_factor_for_mapping_FEniCS_and_OpenFOAM = tol_factor_for_mapping_FEniCS_and_OpenFOAM 

		# Projection setup
		 # Set how you would like the internal FEniCS projections to operate.
		self.projection_setup = {
			'solver_type' : 'default',
			'preconditioner_type' : 'default',
			'form_compiler_parameters' : {
				'type_of_quadrature_degree' : 'auto',
			},
		}
		self.projection_setup.update(projection_setup)

		# Create all problem setup
		if create_all_problem_setup == True:
			if action_to_do_to_create_all_problem_setup == 'overwrite whatever needed':
				pass

			elif action_to_do_to_create_all_problem_setup == 'remove only the initial guess folder and overwrite whatever needed':

				if utils.checkIfFileExists(self.problem_folder) == True:
					sorted_number_folders = utils.findFoamVariableFolders(self.problem_folder)
					if len(sorted_number_folders) > 0:
						first_time_step_name = sorted_number_folders[0]
						utils.removeFolderIfItExists("%s/%s" %(self.problem_folder, first_time_step_name))

			elif action_to_do_to_create_all_problem_setup == 'remove previous problem folder':
				utils.removeFolderIfItExists(self.problem_folder)

			elif action_to_do_to_create_all_problem_setup == 'remove all previous files':
				utils.removeFilesInFolderAndSubfoldersIfItExists(self.problem_folder)

			elif action_to_do_to_create_all_problem_setup == 'rename previous problem':
				utils.renameFolderIfItExists(self.problem_folder, new_name_type = 'bak')

			else:
				raise ValueError(" ❌ ERROR: action_to_do_to_create_all_problem_setup == '%s' is not defined!" %(action_to_do_to_create_all_problem_setup))

		# Location of the solver

		# Some default setup
		parameters.setdefault('solver', {})
		parameters['solver'].setdefault('type', 'openfoam')

		if parameters['solver']['type'] == 'openfoam':

			# Some default setup
			parameters['solver'].setdefault('openfoam', {})
			parameters['solver']['openfoam'].setdefault('name', 'simpleFoam')

			# Location of the OpenFOAM solver
			self.location_solver = utils.getOpenFOAMSolverLocation(parameters['solver']['openfoam']['name'])

		elif parameters['solver']['type'] == 'custom':

			if ('custom' not in parameters['solver']) or ('name' not in parameters['solver']['custom']) or ('location' not in parameters['solver']['custom']):
				raise ValueError(""" ❌ ERROR: Forgot to set the custom solver? Its name should be defined in parameters['solver']['custom']['name'] (<-> name of the solver that can be called in the OpenFOAM environment), together with its location in parameters['solver']['custom']['location'] (<-> path to its source files)!

The solver can be set in any of the two options below:

1) For a solver that you have programmed:
 - parameters['solver']['type'] = 'custom'
 - parameters['solver']['custom']['name'] = ??? (<-> name of the solver that can be called in the OpenFOAM environment)
 - parameters['solver']['custom']['location'] = ??? (<-> path to its source files)
and set by parameters['compile_modules_if_needed'] as: True (i.e., compile whenever needed, such as in the situations where it is still not compiled, or the source files are newer than the compilation) or False (i.e., do not compile whenever needed -- If the solver still has not been compiled or it is outdated with respect to the source files, you will have to compile it externally). It can also be highlighted that parameters['compile_modules_if_needed'] is applied to any OpenFOAM C++ modules/libraries that are included in FEniCS TopOpt Foam.

2) If you use the setup below, you simply won't have the option parameters['compile_modules_if_needed'] applied to the solver. In essence, that's the only difference with respect to the 'custom' option.
 - parameters['solver']['type'] = 'openfoam'
 - parameters['solver']['openfoam']['name'] = ??? (<-> name of the solver that can be called in the OpenFOAM environment)

""")

			# Location of the custom solver
			self.location_solver = "%s/%s" % (parameters['solver']['custom']['location'], parameters['solver']['custom']['name'])

		else:
			raise ValueError(" ❌ ERROR: parameters['solver']['type'] == '%s' is not defined!" %(parameters['solver']['type']))

		# Domain type

		# Some default setup
		parameters.setdefault('domain type', '2D')

		domain_type = parameters['domain type']
		self.domain_type = domain_type

		# Create the OpenFOAM mesh from the FEniCS mesh
		if use_mesh_from_foam_solver == False:
			createMeshFromFEniCS(mesh, boundary_data, problem_folder, domain_type = domain_type, python_write_precision = python_write_precision)
			parameters.setdefault('mesh', {})
			parameters['mesh']['type'] = 'OpenFOAM mesh' # Set FoamSolver to load the mesh that was prepared above

		# Check if configurations_dictionary is probably OK
		assert 'controlDict' in configurations_dictionary, " ❌ ERROR: Forgot to define 'controlDict'."
		assert 'fvSchemes' in configurations_dictionary, " ❌ ERROR: Forgot to define 'fvSchemes'."
		assert 'fvSolution' in configurations_dictionary, " ❌ ERROR: Forgot to define 'fvSolution'."
		configurations_dictionary['controlDict'].setdefault('writeFormat', 'ascii')
		assert configurations_dictionary['controlDict']['writeFormat'] == 'ascii', " ❌ ERROR: This code requires that configurations_dictionary['controlDict']['writeFormat'] == 'ascii'. Please change '%s' to 'ascii'" %(configurations_dictionary['controlDict']['writeFormat'])

		# Create the problem folders
		if create_all_problem_setup == True:
			self._createInitialFiles(properties_dictionary, configurations_dictionary, python_write_precision = python_write_precision)
		
		# Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

		#### Optimize the mesh for OpenFOAM

		self.optimizeMeshForOpenFOAM()

		#### Set FoamSolver

		# FEniCS mesh
		self.mesh = mesh

		# Create the FoamSolver
		self.foam_solver = FoamSolver(parameters = parameters, python_write_precision = python_write_precision)

		# Prepare for editing in step 0
		self.foam_solver.prepareForEditing(time_step_name = '0')

		# Workaround for having all the configurations available in self.foam_solver,
		 # because FoamReader can't read configurations yet
		if create_all_problem_setup == True:
			for i in range(len(self.foam_solver.foam_configurations)):
				configuration_name = self.foam_solver.foam_configurations[i].name

				if configuration_name in self.configuration_workaround_while_FoamReader_cant_read_configurations:
					configuration_data = self.configuration_workaround_while_FoamReader_cant_read_configurations[configuration_name]
					self.foam_solver.foam_configurations[i].reloadData(configuration_data, configuration_name, set_foam_file_info = True)
					# self.foam_solver.foam_configurations[i].set_to_apply_changes('insert') # Not doing this, because WE KNOW that the configurations are the same of the files

					# If the 'decomposeParDict' has been defined, it is assumed that we want to run the code in parallel
					if configuration_name == 'decomposeParDict':
						self.foam_solver.setToRunInParallel(configuration_data, set_new_data = False)

		# Prepare FunctionSpaces and maps
		self.prepareFunctionSpacesAndMaps()

		# FoamVectors
		foam_vectors = self.foam_solver.getFoamVectors()

		# Dictionary of variables
		self.variable_dictionary = {}
		for i in range(len(foam_vectors)):
			foam_vector = foam_vectors[i]
			self.variable_dictionary[foam_vector.name] = {
				'FoamVector' : foam_vector,
				'FEniCS Function' : None,
			}

		# Additional properties
		self.additional_properties = {}

	####################### optimizeMeshForOpenFOAM ########################

	@utils_fenics.no_annotations()
	def optimizeMeshForOpenFOAM(self, problem_folder = ""):
		"""
		Optimize the mesh for OpenFOAM.
		"""

		if problem_folder == "":
			problem_folder = self.problem_folder

		# Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

		#### Optimize the mesh for OpenFOAM
		if utils_fenics_mpi.runningInSerialOrFirstProcessor():

			with utils_fenics_mpi.first_processor_lock():

				# Some OpenFOAM post-processing utilities assume that, inside the 'constant'
				 # folder, the only folders that are present are folders containing the mesh. This means that
				 # we can not leave a backup of the mesh inside the 'constant' folder. Below, we are copying it
				 # to '[problem_folder]/bak_polyMesh_orig'
				utils.customPrint("\n 🌀 Creating a copy of the current state of the OpenFOAM mesh to %s/bak_polyMesh_orig..." %(problem_folder))
				utils.run_command_in_shell("/bin/cp -pr %s/constant/polyMesh %s/bak_polyMesh_orig" %(problem_folder, problem_folder), mode = 'print directly to terminal', include_delimiters_for_print_to_terminal = False, indent_print = True)

				# Reorder the cells in a way that it is more optimized for OpenFOAM to access it (* Renumber the cell list in order to reduce the bandwidth, reading and renumbering all fields from all the time directories. )
				 # https://cfd.direct/openfoam/user-guide/v7-mesh-description/
				 # https://www.cfd-online.com/Forums/openfoam/95858-owner-neighbor.html
				 # ("Finite Volume Method  A Crash introduction - Wolf Dynamics", p. 92)
				 # "speed-up" ("Finite Volume Method  A Crash introduction - Wolf Dynamics", p. 101)
				 # https://www.openfoam.com/documentation/guides/latest/doc/tutorial-pimplefoam-ami-rotating-fan.html
				utils.customPrint("\n 🌀 Restructuring the mesh for better calculation performance... (i.e., optimize the order of the cells in the OpenFOAM mesh)")
				utils.run_command_in_shell("renumberMesh -case %s -overwrite" %(problem_folder), mode = 'print directly to terminal', indent_print = True)

				# Check the mesh
				if utils.print_level == 'debug':
					utils.customPrint("\n 🌀 Checking if the mesh has been created correctly...")
					utils.run_command_in_shell("checkMesh -case %s -constant -time constant" %(problem_folder), mode = 'print directly to terminal', indent_print = True)

		# Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

	######################### _createInitialFiles ##########################

	@utils_fenics.no_annotations()
	def _createInitialFiles(self, properties_dictionary, configurations_dictionary, python_write_precision = 6):
		"""
		Create initial files for OpenFOAM.
		"""

		utils.customPrint("\n 🌊 Creating initial files for OpenFOAM...")

		# FoamWriter
		foam_writer = FoamWriter(self.problem_folder)

		# FoamReader
		foam_reader = FoamReader(self.problem_folder)

		################################################################
		######################### Variables ############################
		################################################################

		##################### Folder structure #########################

		foam_writer.create_folder_structure()

		################ Variable names from solver ####################

		# File with the variable definitions in the solver
		variable_definition_file = "%s/createFields.H" %(self.location_solver)

		# Recognized variable types
		variable_types = ['volScalarField', 'volVectorField']

		# Get all names of the MUST_READ variables of the above types
		(variable_names_from_solver, variable_types_from_solver) = utils.getMustReadVariableNamesFromSolverFile(variable_definition_file, variable_types, return_types = True)

		foam_variables = []
		for i in range(len(variable_names_from_solver)):
			foam_variables += [{'name' : variable_names_from_solver[i], 'type' : variable_types_from_solver[i]}]

		# The only variables 'caught' here are the ones present in 'createFields.H'.
		 # For any other dependent variable, such as turbulent variables, please use the 'fenics_foam_solver.initFoamVector' function to initialize them. (* Small observation: DO NOT use 'fenics_foam_solver._initFoamVectorFile', because it is meant to be an internal function.)

		#################### Boundary conditions #######################
		# Set some default values. The user should change them at a later part of the code!

		mesh_boundary_data = foam_reader.readMeshData('boundary')['boundary']

		# Initialize the necessary variables
		time_step_name = '0'
		for foam_variable in foam_variables:

			foam_vector_name = foam_variable['name']
			foam_vector_type = foam_variable['type']

			# Initialize the FoamVector file
			self._initFoamVectorFile(foam_vector_name, foam_vector_type, time_step_name = time_step_name, mesh_boundary_data = mesh_boundary_data, foam_writer = foam_writer)

		################################################################
		######################## Properties ############################
		################################################################

		for property_file_name in properties_dictionary:

			data_to_add = {}

			# FoamFile definition
			data_to_add['FoamFile'] = {
				'version' : '2.0',
				'format' : 'ascii',
				'class' : 'dictionary',
				'location' : '"constant"',
				'object' : property_file_name,
			}

			# Add the properties
			for key in properties_dictionary[property_file_name]:
				data_to_add[key] = properties_dictionary[property_file_name][key]

			# Write the data to property file
			foam_writer.writeDataToFile(
				data_to_add = data_to_add, 
				file_to_use = "%s/constant/%s" % (self.problem_folder, property_file_name)
			)

		################################################################
		###################### Configurations ##########################
		################################################################

		assert 'controlDict' in configurations_dictionary
		assert 'fvSchemes' in configurations_dictionary
		assert 'fvSolution' in configurations_dictionary
		#assert 'fvOptions' in configurations_dictionary

		# Workaround for having all the configurations available in self.foam_solver,
		 # because FoamReader can't read configurations yet
		self.configuration_workaround_while_FoamReader_cant_read_configurations = {}

		for configuration_file_name in configurations_dictionary:

			data_to_add = {}

			# FoamFile definition
			data_to_add['FoamFile'] = {
				'version' : '2.0',
				'format' : 'ascii',
				'class' : 'dictionary',
				'location' : '"system"',
				'object' : configuration_file_name,
			}

			# Add the configurations
			for key in configurations_dictionary[configuration_file_name]:
				data_to_add[key] = configurations_dictionary[configuration_file_name][key]

			# Write the data to configuration file
			foam_writer.writeDataToFile(
				data_to_add = data_to_add, 
				file_to_use = "%s/system/%s" % (self.problem_folder, configuration_file_name)
			)

			self.configuration_workaround_while_FoamReader_cant_read_configurations[configuration_file_name] = data_to_add

	######################## getFoamVectorFromName #########################

	def getFoamVectorFromName(self, foam_vector_name):
		"""
		Gets a FoamVector from its name.
		"""
		return self.variable_dictionary[foam_vector_name]['FoamVector']

	############################ initFoamVector ############################

	def initFoamVector(self, foam_vector_name, foam_vector_type, time_step_name = '0', skip_if_exists = False):
		"""
		Initializes a new FoamVector (i.e., creates the file and the FoamVector object).
		-> Use this function if you want to include more variables for OpenFOAM to use.
		"""

		if foam_vector_name in self.variable_dictionary:
			if skip_if_exists == True:
				utils.customPrint(" ❗ Skipping FoamVector '%s', which is already defined!..." %(foam_vector_name))
				return
			else:
				raise ValueError(" ❌ ERROR: FoamVector '%s' is already defined!" %(foam_vector_name))

		# Initializes (i.e., creates) a FoamVector file
		self._initFoamVectorFile(foam_vector_name, foam_vector_type, time_step_name = time_step_name)

		# Prepares a new FoamVector from file
		self.foam_solver.prepareNewFoamVector(foam_vector_name, time_step_name = time_step_name)

		# New FoamVector
		new_foam_vector = self.foam_solver.foam_vectors[len(self.foam_solver.foam_vectors) - 1]

		# Include in the dictionary of variables
		self.variable_dictionary[new_foam_vector.name] = {
			'FoamVector' : new_foam_vector,
			'FEniCS Function' : None,
		}

	######################### _initFoamVectorFile ##########################

	def _initFoamVectorFile(self, foam_vector_name, foam_vector_type, time_step_name = '0', mesh_boundary_data = None, foam_writer = None):
		"""
		Initializes (i.e., creates) a FoamVector file.
		"""

		# FoamWriter
		if type(foam_writer).__name__ == 'NoneType':
			foam_writer = self.foam_solver.foam_writer

		# Boundary data
		if type(mesh_boundary_data).__name__ == 'NoneType':
			mesh_boundary_data = self.foam_solver.foam_mesh.boundaries()

		# Boundary names
		boundary_names = list(mesh_boundary_data.keys())

		# Type check
		if foam_vector_type in ['volScalarField', 'volVectorField']:
			pass
		else:
			raise ValueError(" ❌ ERROR: foam_vector_type == '%s' is not defined!" %(foam_vector_type))

		data_to_add = {}

		# FoamFile definition
		data_to_add['FoamFile'] = {
			'version' : '2.0',
			'format' : 'ascii',
			'class' : foam_vector_type,
			'location' : time_step_name,
			'object' : foam_vector_name,
		}

		# Dimensions
		data_to_add['dimensions'] = self.getFoamMeasurementUnit(foam_vector_name)

		# Set uniform internal fields
		# Some dummy values to change later. Based on pitzDaily OpenFOAM example
		if foam_vector_name == 'k':
			data_to_add['internalField'] = np.array([0.375], dtype = 'float')

		elif foam_vector_name == 'epsilon':
			data_to_add['internalField'] = np.array([14.855], dtype = 'float')

		elif foam_vector_name == 'omega':
			data_to_add['internalField'] = np.array([440.15], dtype = 'float')

		elif foam_vector_name == 'v2':
			data_to_add['internalField'] = np.array([0.25], dtype = 'float')

		else:
			if foam_vector_type == 'volScalarField':
				data_to_add['internalField'] = np.array([0], dtype = 'float')
			elif foam_vector_type == 'volVectorField':
				data_to_add['internalField'] = np.array([0, 0, 0], dtype = 'float')

		data_to_add['boundaryField'] = {}
		for boundary_name in boundary_names:

			data_to_add['boundaryField'][boundary_name] = {}

			########## Default boundary conditions ##########
			# Some dummy values to change later. Based on pitzDaily OpenFOAM example
			# * Just remember: 'frontAndBackSurfaces', 'frontSurface' and 'backSurface' are RESERVED names.
			#   Do not use them anywhere else!

			if boundary_name == 'frontAndBackSurfaces':

				# Add the necessary boundary conditions that enable 2D or 2D axisymmetric simulation
				 # https://www.cfd-online.com/Forums/openfoam-solving/60539-2d-axisymmetric-swirl.html
				if self.domain_type == '2D':
					data_to_add['boundaryField']['frontAndBackSurfaces']['type'] = 'empty'
				elif self.domain_type == '2D axisymmetric':
					pass
				else:
					raise ValueError(" ❌ ERROR: self.domain_type == '%s' is using a 2D/2D axisymmetric boundary condition ('%s')???" %(self.domain_type, boundary_name))

			elif boundary_name == 'frontSurface':

				# Add the necessary boundary conditions that enable 2D or 2D axisymmetric simulation
				 # https://www.cfd-online.com/Forums/openfoam-solving/60539-2d-axisymmetric-swirl.html
				if self.domain_type == '2D':
					raise ValueError(" ❌ ERROR: self.domain_type == '%s' is using a 2D/2D axisymmetric boundary condition ('%s')???" %(self.domain_type, boundary_name))
				elif self.domain_type == '2D axisymmetric':
					data_to_add['boundaryField']['frontSurface']['type'] = 'wedge'
				else:
					raise ValueError(" ❌ ERROR: self.domain_type == '%s' is using a 2D/2D axisymmetric boundary condition ('%s')???" %(self.domain_type, boundary_name))

			elif boundary_name == 'backSurface':

				# Add the necessary boundary conditions that enable 2D or 2D axisymmetric simulation
				 # https://www.cfd-online.com/Forums/openfoam-solving/60539-2d-axisymmetric-swirl.html
				if self.domain_type == '2D':
					raise ValueError(" ❌ ERROR: self.domain_type == '%s' is using a 2D/2D axisymmetric boundary condition ('%s')???" %(self.domain_type, boundary_name))
				elif self.domain_type == '2D axisymmetric':
					data_to_add['boundaryField']['backSurface']['type'] = 'wedge'
				else:
					raise ValueError(" ❌ ERROR: self.domain_type == '%s' is using a 2D/2D axisymmetric boundary condition ('%s')???" %(self.domain_type, boundary_name))

			else:

				if mesh_boundary_data[boundary_name]['type'] in ['cyclic', 'cyclicAMI', 'cyclicACMI']:
					# Set periodic ('cyclic', 'cyclicAMI', 'cyclicACMI') boundary condition
					data_to_add['boundaryField'][boundary_name]['type'] = mesh_boundary_data[boundary_name]['type']

				else:

					if foam_vector_name == 'p':
						if utils.strIsIn('inlet', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'zeroGradient'
						elif utils.strIsIn('outlet', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'fixedValue'
							data_to_add['boundaryField'][boundary_name]['value'] = np.array([0], dtype = 'float')
						elif utils.strIsIn('wall', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'zeroGradient'

					elif foam_vector_name == 'U' or foam_vector_name == 'Urel':
						if utils.strIsIn('inlet', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'fixedValue'
							data_to_add['boundaryField'][boundary_name]['value'] = np.array([1, 0, 0], dtype = 'float')
						elif utils.strIsIn('outlet', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'zeroGradient'
						elif utils.strIsIn('wall', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'noSlip'

					elif foam_vector_name == 'T':
						if utils.strIsIn('inlet', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'zeroGradient'
						elif utils.strIsIn('outlet', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'fixedValue'
							data_to_add['boundaryField'][boundary_name]['value'] = np.array([0], dtype = 'float')
						elif utils.strIsIn('wall', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'zeroGradient'

					elif foam_vector_name == 'k':
						if utils.strIsIn('inlet', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'fixedValue'
							data_to_add['boundaryField'][boundary_name]['value'] = np.array([0.375], dtype = 'float')
						elif utils.strIsIn('outlet', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'zeroGradient'
						elif utils.strIsIn('wall', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'kqRWallFunction'
							data_to_add['boundaryField'][boundary_name]['value'] = np.array([0.375], dtype = 'float')

					elif foam_vector_name == 'epsilon':
						if utils.strIsIn('inlet', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'fixedValue'
							data_to_add['boundaryField'][boundary_name]['value'] = np.array([14.855], dtype = 'float')
						elif utils.strIsIn('outlet', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'zeroGradient'
						elif utils.strIsIn('wall', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'epsilonWallFunction'
							data_to_add['boundaryField'][boundary_name]['value'] = np.array([14.855], dtype = 'float')

					elif foam_vector_name == 'omega':
						if utils.strIsIn('inlet', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'fixedValue'
							data_to_add['boundaryField'][boundary_name]['value'] = np.array([440.15], dtype = 'float')
						elif utils.strIsIn('outlet', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'zeroGradient'
						elif utils.strIsIn('wall', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'omegaWallFunction'
							data_to_add['boundaryField'][boundary_name]['value'] = np.array([440.15], dtype = 'float')

					elif foam_vector_name == 'nut':
						if utils.strIsIn('wall', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'nutkWallFunction'
							data_to_add['boundaryField'][boundary_name]['value'] = np.array([0], dtype = 'float')
						else:
							data_to_add['boundaryField'][boundary_name]['type'] = 'calculated'
							data_to_add['boundaryField'][boundary_name]['value'] = np.array([0], dtype = 'float')

					elif foam_vector_name == 'nuTilda':
						if utils.strIsIn('inlet', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'nutkWallFunction'
							data_to_add['boundaryField'][boundary_name]['value'] = np.array([0], dtype = 'float')
						elif utils.strIsIn('outlet', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'zeroGradient'
						elif utils.strIsIn('wall', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'zeroGradient'

					elif foam_vector_name == 'v2':
						if utils.strIsIn('inlet', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'fixedValue'
							data_to_add['boundaryField'][boundary_name]['value'] = np.array([0.25], dtype = 'float')
						elif utils.strIsIn('outlet', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'zeroGradient'
						elif utils.strIsIn('wall', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'v2WallFunction'
							data_to_add['boundaryField'][boundary_name]['value'] = np.array([0.25], dtype = 'float')

					elif foam_vector_name == 'f':
						if utils.strIsIn('inlet', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'zeroGradient'
						elif utils.strIsIn('outlet', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'zeroGradient'
						elif utils.strIsIn('wall', boundary_name, ignore_case = False):
							data_to_add['boundaryField'][boundary_name]['type'] = 'fWallFunction'
							data_to_add['boundaryField'][boundary_name]['value'] = np.array([0.25], dtype = 'float')

					elif foam_vector_name == 'alpha_design':
							# The design variable is NOT solved by OpenFOAM, which means that OpenFOAM should not try to impose any boundary condition on it.
							# https://www.openfoam.com/documentation/user-guide/standard-boundaryconditions.php
						data_to_add['boundaryField'][boundary_name]['type'] = 'calculated'
						data_to_add['boundaryField'][boundary_name]['value'] = np.array([0], dtype = 'float')

					elif foam_vector_name.startswith('U_'):
							# This variable is NOT solved by OpenFOAM, which means that OpenFOAM should not try to impose any boundary condition on it.
							# https://www.openfoam.com/documentation/user-guide/standard-boundaryconditions.php
						data_to_add['boundaryField'][boundary_name]['type'] = 'calculated'
						data_to_add['boundaryField'][boundary_name]['value'] = np.array([0, 0, 0], dtype = 'float')

					else:
						# This error is just because I want to guarantee that the variable has a definition here (in this function). Also, the variable needs to be in self.dictionary_measurement_units.
						utils.customPrint(" ❗ Variable '%s' does not have a predefined sample value. Setting sample value to a zero 'fixedValue'..." %(foam_vector_name))
						#raise ValueError(" ❌ ERROR: foam_vector_name == '%s' is not defined!" %(foam_vector_name))
				
					if len(data_to_add['boundaryField'][boundary_name]) == 0:
						if foam_vector_type == 'volScalarField':
							data_to_add['boundaryField'][boundary_name]['type'] = 'fixedValue'
							data_to_add['boundaryField'][boundary_name]['value'] = np.array([0], dtype = 'float')

						elif foam_vector_type == 'volVectorField':
							data_to_add['boundaryField'][boundary_name]['type'] = 'fixedValue'
							data_to_add['boundaryField'][boundary_name]['value'] = np.array([0, 0, 0], dtype = 'float')

		# Final check
		if self.domain_type == '2D':
			assert 'frontAndBackSurfaces' in data_to_add['boundaryField']
		elif self.domain_type == '2D axisymmetric':
			assert 'frontSurface' in data_to_add['boundaryField']
			assert 'backSurface' in data_to_add['boundaryField']

		# Write the data to variable file
		foam_writer.writeDataToFile(
			data_to_add = data_to_add, 
			file_to_use = "%s/%s/%s" % (self.problem_folder, time_step_name, foam_vector_name)
		)

	####################### getFoamMeasurementUnit #########################

	@utils_fenics.no_annotations()
	def getFoamMeasurementUnit(self, foam_name):
		"""
		Returns the measurement unit code of a variable in OpenFOAM.
		"""

		if foam_name not in self.dictionary_measurement_units:

			# Considering "starred" names, such as 'U_*'

			names_without_star = [name[:-1] for name in self.dictionary_measurement_units if name.endswith('*')]

			found = False
			for name_without_star in names_without_star:
				if foam_name.startswith(name_without_star):
					name_with_star = name_without_star + "*"
					unit_name = self.dictionary_measurement_units[name_with_star]
					found = True
					break

			if found == False:
				raise ValueError(""" ❌ ERROR: Variable '%s' does not exist in the dictionary of measurement units (%s)!
But: You can add it through the keyword parameter 'additional_measurement_units' in FEniCSFoamSolver([...]).
For example,

fenics_foam_solver = FEniCSFoamSolver(
	[...],
	additional_measurement_units = {
		'Unew' : 'velocity',             # String name
		'xyz'  : [0, 2, -1, 0, 0, 0, 0], # Specified in the OpenFOAM format
	}
)

Check "utils/utils.py > convertToFoamUnitSpecification" in order to know which 
measurement units are already available through string names. If the desired
measurement unit is not defined there, you need to specify a Python list
in the OpenFOAM format ( https://cfd.direct/openfoam/user-guide/v7-basic-file-format/ ),
as shown above for 'xyz'.
""" %(foam_name, self.dictionary_measurement_units))

		else:
			unit_name = self.dictionary_measurement_units[foam_name]

		return utils.convertToFoamUnitSpecification(unit_name)

	#################### prepareFunctionSpacesAndMaps ######################

	@utils_fenics.no_annotations()
	def prepareFunctionSpacesAndMaps(self):
		"""
		Prepare the FunctionSpaces and DoF maps for FEniCS and the OpenFOAM variables.
		"""

		utils.initTimeCount('Creating variable maps')

		utils.customPrint("\n 🌊 Preparing function spaces and maps...")

		# Create DoF maps -- Assuming triangular mesh (for the 2D axisymmetric domain)
		foam_dof_coordinates = self.foam_solver.foam_mesh.cell_centers(compute_2D_cell_centers_for_triangles_if_2D_axisymmetric = True)

		#### Scalar values

		# For scalar values (0 dimensions)

		utils.initTimeCount('Creating variable map for scalar values (0 dimensions)')

		utils.customPrint("\n 🌊 Creating map for scalar values (0 dimensions)...")

		# DG0 FunctionSpace
		DG0_function_space = utils_fenics.createDG0FEniCSFunctionSpace(self.mesh, dim = 0)

		# DoF coordinates
		fenics_dof_coordinates = utils_fenics.getDoFCoordinatesFromFunctionSpace(DG0_function_space)

		if utils_fenics_mpi.runningInParallel():

			# Save for later
			fenics_dof_coordinates_orig = fenics_dof_coordinates

			# Gather the DoF coordinates
			(fenics_dof_coordinates_gathered, map_local_to_global_DoFs, map_global_to_local_DoFs) = utils_fenics_mpi.getGatheredDoFCoordinates(DG0_function_space, return_local_maps = True)

			# Use the gathered DoF coordinates
			fenics_dof_coordinates = fenics_dof_coordinates_gathered

		# Minimum element size
		hmin = self.mesh.hmin()
		if utils_fenics_mpi.runningInParallel():
			hmin = utils_fenics_mpi.evaluateBetweenProcessors(hmin, operation = 'minimum', proc_destination = 'all')

		# Create the DoF maps
		(self.map_DoFs_foam_to_fenics, self.map_DoFs_fenics_to_foam) = utils_fenics.generateDoFMapsOpenFOAM_FEniCS(fenics_dof_coordinates, foam_dof_coordinates, self.domain_type, mesh_hmin = hmin, tol_factor = self.tol_factor_for_mapping_FEniCS_and_OpenFOAM)

		if utils_fenics_mpi.runningInParallel():

			# Maps: Global FEniCS <-> Global FoamVector
			map_DoFs_global_foam_to_global_fenics = self.map_DoFs_foam_to_fenics
			map_DoFs_global_fenics_to_global_foam = self.map_DoFs_fenics_to_foam

			# Maps: Local FEniCS <-> Global FoamVector
			map_DoFs_global_foam_to_local_fenics = [map_DoFs_global_fenics_to_global_foam[map_local_to_global_DoFs[i_fenics_local]] for i_fenics_local in range(len(map_local_to_global_DoFs))]
			map_DoFs_local_fenics_to_global_foam = utils_fenics.invertMapOrder(map_DoFs_global_foam_to_local_fenics)

			# Maps: Local FEniCS <-> Local FoamVector
			 # The local mapping between FEniCS and FoamVector will 
			 # be selected as the same (for simplicity).
			map_DoFs_local_fenics_to_local_foam = np.arange(0,len(map_DoFs_local_fenics_to_global_foam))
			map_DoFs_local_foam_to_local_fenics = map_DoFs_local_fenics_to_local_foam.copy()

			# Save all maps
			self.map_DoFs_global_foam_to_global_fenics = map_DoFs_global_foam_to_global_fenics
			self.map_DoFs_global_fenics_to_global_foam = map_DoFs_global_fenics_to_global_foam
			self.map_DoFs_local_fenics_to_global_foam = map_DoFs_local_fenics_to_global_foam
			self.map_DoFs_global_foam_to_local_fenics = map_DoFs_global_foam_to_local_fenics
			self.map_DoFs_local_fenics_to_local_foam = map_DoFs_local_fenics_to_local_foam
			self.map_DoFs_local_foam_to_local_fenics = map_DoFs_local_foam_to_local_fenics

			# Use the local maps
			self.map_DoFs_foam_to_fenics = map_DoFs_local_foam_to_local_fenics
			self.map_DoFs_fenics_to_foam = map_DoFs_local_fenics_to_local_foam

			# Set the local DoF mapping for FoamMesh, in order for FoamVector
			 # to be able to be updated in parallel.
			self.foam_solver.foam_mesh.set_local_mapping(self.map_DoFs_global_foam_to_local_fenics, self.map_DoFs_local_fenics_to_global_foam)

			# Back to the local DoF coordinates
			fenics_dof_coordinates = fenics_dof_coordinates_orig

		else:
			pass

		utils.finishTimeCount('Creating variable map for scalar values (0 dimensions)')

		#### Vector values

		if self.domain_type == '2D': # For FEniCS vector values (2 dimensions)

			utils.initTimeCount('Creating variable map for vector values (2 dimensions)')

			utils.customPrint("\n 🌊 Creating map for vector values (2 dimensions)...")

			# DG0 VectorFunctionSpace with 2 dimensions
			DG0_function_space_2 = utils_fenics.createDG0FEniCSFunctionSpace(self.mesh, dim = 2)

			# Create the DoF maps
			(map_from_component_to_function, map_from_function_to_component) = utils_fenics.getDoFmapFromFEniCSVectorFunctiontoComponent(DG0_function_space_2)

			# Save the DoF maps
			self.maps_from_component_to_function = {
				'2 components' : {
					'from component to function' : map_from_component_to_function,
					'from function to component' : map_from_function_to_component,
				},
			}

			utils.finishTimeCount('Creating variable map for vector values (2 dimensions)')

		elif self.domain_type == '2D axisymmetric' or self.domain_type == '3D': # For FEniCS vector values (3 dimensions)

			utils.initTimeCount('Creating variable map for vector values (3 dimensions)')

			utils.customPrint("\n 🌊 Creating map for vector values (3 dimensions)...")

			# DG0 VectorFunctionSpace with 3 dimensions
			DG0_function_space_3 = utils_fenics.createDG0FEniCSFunctionSpace(self.mesh, dim = 3)

			# Create the DoF maps
			(map_from_component_to_function, map_from_function_to_component) = utils_fenics.getDoFmapFromFEniCSVectorFunctiontoComponent(DG0_function_space_3)

			# Save the DoF maps
			self.maps_from_component_to_function = {
				'3 components' : {
					'from component to function' : map_from_component_to_function,
					'from function to component' : map_from_function_to_component,
				},
			}

			utils.finishTimeCount('Creating variable map for vector values (3 dimensions)')

		else:
			raise ValueError(" ❌ ERROR: self.domain_type == '%s' is not defined!" %(self.domain_type))

		utils.finishTimeCount('Creating variable maps')

	#################### setFEniCSFunctionToFoamVector #####################

	@utils_fenics.no_annotations()
	def setFEniCSFunctionToFoamVector(self, 
		fenics_function, 
		foam_variable_name = "", 
		convert_to_usual_units_if_necessary = True, 
		set_calculated_foam_boundaries = False, 
		convert_from_usual_units_if_necessary = True, 
		ensure_maximum_minimum_values_after_projection = False,
		tol_for_maximum_minimum_values_after_projection = 1.E-20
		):
		"""
		Sets a FEniCS Function to the "internalField" of an OpenFOAM variable (* The "boundaryField" is not considered in this operation (at least for now), because "boundaryField" is closely related to the imposition of the boundary conditions. Check "fenics_foam_solver.getFoamVectorBoundaryValues".)
		* DO NOT use 'Indexed' variables that you obtain by u[num_global_component] or split(u)[num_local_component].
		  The only variables recognized here are u.split()[num_local_component] or u.split(deepcopy = True)[num_local_component],
		  which effectively create copies of the array of values.
		"""

		utils.customPrint("\n 🌊 Setting OpenFOAM variable '%s' from FEniCS..." %(foam_variable_name))

		# Check the type
		fenics_function_orig = fenics_function

		# Function
		if type(fenics_function_orig).__name__ == 'Function':
			pass

		# Indexed
		elif type(fenics_function_orig).__name__ == 'Indexed':
			raise NotImplementedError(""" ❌ ERROR: Not implemented for type(fenics_function_orig).__name__ == '%s'! 
DO NOT use 'Indexed' variables that you obtain by u[num_global_component] or split(u)[num_local_component].
The only variables recognized here are u.split()[num_local_component] or u.split(deepcopy = True)[num_local_component],
which effectively create copies of the array of values.""" %(type(fenics_function_orig).__name__))

		# Constant
		elif type(fenics_function_orig).__name__ == 'Constant':
				
			# FoamVector
			foam_vector = self.variable_dictionary[foam_variable_name]['FoamVector']

			# Local values from the FoamVector
			foam_vector_array = foam_vector.get_local()

			# Converting Constant to float
			float_value = utils_fenics.floatVector(fenics_function_orig)

			# If it is a scalar
			if ('int' in type(float_value).__name__) or ('float' in type(float_value).__name__):
				foam_vector_new_array = np.ones_like(foam_vector_array)*float_value

			else: # If it is a vector

				if self.domain_type == '2D': # Ignore third component

					number_of_elements = foam_vector_array.shape[0]		
					foam_vector_new_array = np.repeat([[float_value[0], float_value[1], 0.0]], number_of_elements, axis = 0)

				elif self.domain_type == '2D axisymmetric' or domain_type == '3D':

					number_of_elements = foam_vector_array.shape[0]		
					foam_vector_new_array = np.repeat([float_value], number_of_elements, axis = 0)

				else:
					raise ValueError(" ❌ ERROR: self.domain_type == %s is not defined here!" %(self.domain_type))

			# Set the local changes
			foam_vector.set_local(foam_vector_new_array)

			# Set to apply the changes to file
			foam_vector.set_to_apply_changes('insert')

			return

		else:
			raise NotImplementedError(" ❌ ERROR: Not implemented for type(fenics_function_orig).__name__ == '%s'! Please, consider using a Function or a Constant!" %(type(fenics_function_orig).__name__))

		# Check if the variable name exists in OpenFOAM
		if foam_variable_name not in self.variable_dictionary:
			raise ValueError(" ❌ ERROR: foam_variable_name == '%s' not available in self.variable_dictionary = %s!" %(foam_variable_name, self.variable_dictionary))

		# Create the necessary DG0 function for the OpenFOAM variable (if not already created)
		if type(self.variable_dictionary[foam_variable_name]['FEniCS Function']).__name__ == 'NoneType':

			# Get the number of components of the FEniCS function
			dim_fenics_function = fenics_function.function_space().num_sub_spaces()

			# Create the corresponding DG0 Function
			if dim_fenics_function > 1:
				DG0_function = utils_fenics.createDG0FEniCSFunction(self.mesh, dim = dim_fenics_function)
			else:
				DG0_function = utils_fenics.createDG0FEniCSFunction(self.mesh, dim = 0)

			# Save for later use
			self.variable_dictionary[foam_variable_name]['FEniCS Function'] = DG0_function

		# Get the DG0 Function
		DG0_function = self.variable_dictionary[foam_variable_name]['FEniCS Function']

		# Update the DG0 function
		DG0_function.assign(utils_fenics.configuredProject(fenics_function, DG0_function.function_space(), self.projection_setup, skip_projection_if_same_FunctionSpace = True))
		#DG0_function.assign(project(fenics_function, DG0_function.function_space()))
		#DG0_function.interpolate(fenics_function)
		#DG0_function.assign(fenics_function)

		# Keep maximum/minimum values from before the projection
		if ensure_maximum_minimum_values_after_projection == True:

			# Maximum and minimum values BEFORE projection
			fenics_function_max = fenics_function.vector().max()
			fenics_function_min = fenics_function.vector().min()

			if utils_fenics_mpi.runningInParallel():
				fenics_function_max = utils_fenics_mpi.evaluateBetweenProcessors(fenics_function_max, operation = 'maximum', proc_destination = 'all')
				fenics_function_min = utils_fenics_mpi.evaluateBetweenProcessors(fenics_function_min, operation = 'minimum', proc_destination = 'all')

			# Get all values AFTER projection
			DG0_function_array = DG0_function.vector().get_local()

			# Set the maximum and minimum values
			DG0_function_array[DG0_function_array < fenics_function_min + tol_for_maximum_minimum_values_after_projection] = fenics_function_min
			DG0_function_array[DG0_function_array > fenics_function_max - tol_for_maximum_minimum_values_after_projection] = fenics_function_max

			# Set the new values
			DG0_function.vector().set_local(DG0_function_array)
			DG0_function.vector().apply('insert')

		# Post-processing
		self.postProcessFEniCSVariableToFoam(foam_variable_name, DG0_function, convert_to_usual_units_if_necessary = convert_to_usual_units_if_necessary)

		# FoamVector
		foam_vector = self.variable_dictionary[foam_variable_name]['FoamVector']

		# Transfer the values of the FEniCS variable to the FoamVector
		utils_fenics.FEniCSFunctionToFoamVector(DG0_function, foam_vector, self.map_DoFs_foam_to_fenics, self.maps_from_component_to_function, self.domain_type)

		# I think the part below should not be needed when you have already set the boundary
		 # conditions as 'calculated' or something like that, because the 'value'
		 # is an 'initial guess' for OpenFOAM to compute the value.  
		if set_calculated_foam_boundaries == True:
			self.setAllEqualFoamBoundaryConditions(foam_variable_name, 'calculated', DG0_function, check_consistency_of_imposed_values = False, convert_from_usual_units_if_necessary = convert_from_usual_units_if_necessary)

	#################### setFoamVectorToFEniCSFunction #####################

	@utils_fenics.no_annotations()
	def setFoamVectorToFEniCSFunction(self, 
		fenics_function, foam_variable_name = "", 
		convert_to_usual_units_if_necessary = True, 
		impose_boundaryField = False, 
		foam_vector = None,

		# Filter for smoother transitions after the projection
		apply_filter_for_smoother_transitions = True,#False,
		filter_parameters = {},
		):
		"""
		Sets a FEniCS Function from the "internalField" of an OpenFOAM variable (* The "boundaryField" is not considered in this operation by default, because "boundaryField" is closely related to the imposition of the boundary conditions. Check "fenics_foam_solver.getFoamVectorBoundaryValues".).
		* DO NOT use 'Indexed' variables that you obtain by u[num_global_component] or split(u)[num_local_component].
		  The only variables recognized here are u.split()[num_local_component] or u.split(deepcopy = True)[num_local_component],
		  which effectively create copies of the array of values.
		"""

		utils.customPrint("\n 🌊 Setting OpenFOAM variable '%s' to FEniCS..." %(foam_variable_name))

		# Check the type
		fenics_function_orig = fenics_function

		# Function
		if type(fenics_function_orig).__name__ == 'Function':
			pass

		# Indexed
		elif type(fenics_function_orig).__name__ == 'Indexed':

			raise NotImplementedError(""" ❌ ERROR: Not implemented for type(fenics_function_orig).__name__ == '%s'! 
DO NOT use 'Indexed' variables that you obtain by u[num_global_component] or split(u)[num_local_component].
The only variables recognized here are u.split()[num_local_component] or u.split(deepcopy = True)[num_local_component],
which effectively create copies of the array of values.""" %(type(fenics_function_orig).__name__))

		else:
			raise NotImplementedError(" ❌ ERROR: Not implemented for type(fenics_function_orig).__name__ == '%s'! Please, consider using a Function!" %(type(fenics_function_orig).__name__))

		# Check if the variable name exists in OpenFOAM
		if type(foam_vector).__name__ != 'NoneType':

			using_separated_foam_vector = True

			# Get the number of components of the FEniCS function
			dim_fenics_function = fenics_function.function_space().num_sub_spaces()

			# Create the corresponding DG0 Function
			if dim_fenics_function > 1:
				DG0_function = utils_fenics.createDG0FEniCSFunction(self.mesh, dim = dim_fenics_function)
			else:
				DG0_function = utils_fenics.createDG0FEniCSFunction(self.mesh, dim = 0)

			# Save for later use
			self.variable_dictionary[foam_variable_name]['FEniCS Function'] = DG0_function

		elif foam_variable_name in self.variable_dictionary:

			using_separated_foam_vector = False

			# FoamVector
			foam_vector = self.variable_dictionary[foam_variable_name]['FoamVector']

			# Create the necessary DG0 function for the OpenFOAM variable
			if type(self.variable_dictionary[foam_variable_name]['FEniCS Function']).__name__ == 'NoneType':

				# Get the number of components of the FEniCS function
				dim_fenics_function = fenics_function.function_space().num_sub_spaces()

				# Create the corresponding DG0 Function
				if dim_fenics_function > 1:
					DG0_function = utils_fenics.createDG0FEniCSFunction(self.mesh, dim = dim_fenics_function)
				else:
					DG0_function = utils_fenics.createDG0FEniCSFunction(self.mesh, dim = 0)

				# Save for later use
				self.variable_dictionary[foam_variable_name]['FEniCS Function'] = DG0_function

			# DG0 Function
			DG0_function = self.variable_dictionary[foam_variable_name]['FEniCS Function']

		else:
			raise ValueError(" ❌ ERROR: foam_variable_name == '%s' not available in self.variable_dictionary = %s!" %(foam_variable_name, self.variable_dictionary))

		# Transfer the values of the FoamVector to the FEniCS variable
		utils_fenics.FoamVectorToFEniCSFunction(foam_vector, DG0_function, self.map_DoFs_fenics_to_foam, self.maps_from_component_to_function, self.domain_type)

		# Update FEniCS Function from DG0
		fenics_function.assign(utils_fenics.configuredProject(DG0_function, fenics_function.function_space(), self.projection_setup, skip_projection_if_same_FunctionSpace = True))
		#fenics_function.assign(project(DG0_function, fenics_function.function_space()))
		#fenics_function.interpolate(DG0_function)
		#fenics_function.assign(DG0_function)

		# Impose 'boundaryField' if wanted
		if impose_boundaryField == True:
			if using_separated_foam_vector == True:
				self.getFoamVectorBoundaryValuesInFEniCS(foam_variable_name = foam_variable_name, convert_to_usual_units_if_necessary = False, type_of_FEniCS_Function = fenics_function, foam_vector = foam_vector)
			else:
				self.getFoamVectorBoundaryValuesInFEniCS(foam_variable_name = foam_variable_name, convert_to_usual_units_if_necessary = False, type_of_FEniCS_Function = fenics_function)

		# Post-processing
		if using_separated_foam_vector == True:
			pass
		else:
			self.postProcessFEniCSVariableFromFoam(foam_variable_name, fenics_function, convert_to_usual_units_if_necessary = convert_to_usual_units_if_necessary)

		# Apply a filter for smoother transitions
		if apply_filter_for_smoother_transitions == True:

			# Check the type of variable
			if foam_vector.num_components == 1:
				type_of_variable = 'scalar'
			elif foam_vector.num_components == 3:
				type_of_variable = 'vector'
			else:
				raise ValueError(" ❌ ERROR: foam_vector.num_components == %d is not implemented!" %(foam_vector.num_components))

			# Apply filter
			utils_fenics.filterVariable(fenics_function, fenics_function.function_space(), domain_type = self.domain_type, filter_parameters = filter_parameters, overload_variable = True, type_of_variable = type_of_variable)

			# Post-processing
			if using_separated_foam_vector == True:
				pass
			else:
				self.postProcessFEniCSVariableFromFoam(foam_variable_name, fenics_function, convert_to_usual_units_if_necessary = False)

	################# getFoamVectorBoundaryValuesInFEniCS ##################

	@utils_fenics.no_annotations()
	def getFoamVectorBoundaryValuesInFEniCS(self, 
		foam_variable_name = "", 
		convert_to_usual_units_if_necessary = True, 
		type_of_FEniCS_Function = 'DG0', 
		foam_vector = None
		):
		"""
		Converts the "boundaryField" of an OpenFOAM variable to a FEniCS Function (* The "internalField" is considered in 'fenics_foam_solver.setFoamVectorToFEniCSFunction' and 'fenics_foam_solver.setFEniCSFunctionToFoamVector').
		-> 'type_of_FEniCS_Function' can be either:
			String-valued: 'DG0' or 'CG1'	
			FEniCS Function
		"""

		utils.customPrint("\n 🌊 Generating the boundary values of the OpenFOAM variable '%s' to FEniCS..." %(foam_variable_name))

		# Check if FoamVector has been provided
		if type(foam_vector).__name__ != 'NoneType':
			pass

		elif foam_variable_name in self.variable_dictionary: # Check if the variable name exists in OpenFOAM

			# FoamVector
			foam_vector = self.variable_dictionary[foam_variable_name]['FoamVector']

		else:
			raise ValueError(" ❌ ERROR: foam_variable_name == '%s' not available in self.variable_dictionary = %s!" %(foam_variable_name, self.variable_dictionary))

		# Generate the maps if not already generated
		if 'map_foam_faces_to_fenics_cells' not in self.__dict__:

			# Minimum element size
			hmin = self.mesh.hmin()
			if utils_fenics_mpi.runningInParallel():
				hmin = utils_fenics_mpi.evaluateBetweenProcessors(hmin, operation = 'minimum', proc_destination = 'all')

			(self.map_foam_faces_to_fenics_cells, self.map_fenics_cells_to_foam_faces, self.map_foam_faces_to_fenics_cells_boundary_coords) = utils_fenics.generateFacetMapsOpenFOAM_FEniCSCells(self.mesh, self.foam_solver.foam_mesh, self.domain_type, mesh_hmin = hmin, tol_factor = self.tol_factor_for_mapping_FEniCS_and_OpenFOAM)

		# Convert to a FEniCS Function
		fenics_function = utils_fenics.FoamVectorBoundaryToFEniCSFunction(self.mesh, foam_vector, self.foam_solver.foam_mesh, self.map_foam_faces_to_fenics_cells, self.domain_type, type_of_FEniCS_Function = type_of_FEniCS_Function, map_foam_faces_to_fenics_cells_boundary_coords = self.map_foam_faces_to_fenics_cells_boundary_coords)

		# Post-processing
		self.postProcessFEniCSVariableFromFoam(foam_variable_name, fenics_function, convert_to_usual_units_if_necessary = convert_to_usual_units_if_necessary)

		return fenics_function

	################# postProcessFEniCSVariableFromFoam ####################

	def postProcessFEniCSVariableFromFoam(self, foam_variable_name, fenics_function, convert_to_usual_units_if_necessary = True):
		"""
		Post-processing from Foam.
		"""
		return self.__postProcessFEniCSVariable('from Foam', foam_variable_name, fenics_function, convert_to_usual_units_if_necessary = convert_to_usual_units_if_necessary)

	################### postProcessFEniCSVariableToFoam ####################

	def postProcessFEniCSVariableToFoam(self, foam_variable_name, fenics_function, convert_to_usual_units_if_necessary = True):
		"""
		Post-processing to Foam.
		"""
		return self.__postProcessFEniCSVariable('to Foam', foam_variable_name, fenics_function, convert_to_usual_units_if_necessary = convert_to_usual_units_if_necessary)

	##################### __postProcessFEniCSVariable ######################

	def __postProcessFEniCSVariable(self, type_of_post_processing, foam_variable_name, fenics_function, convert_to_usual_units_if_necessary = True):
		"""
		Post-processing necessary for the resulting FEniCS variable. The main points are listed below:

		-> For incompressible flow:
			1) Send the 'rho-normalized pressure' (OpenFOAM) as 'pressure' to FEniCS.
			2) Send the 'pressure' (FEniCS) as 'rho-normalized pressure' to OpenFOAM.
			3) Threshold some values in order to guarantee that the projection performed
			   from or to DG0 does not inadvertently return non-physical negative values.
		"""

		if type(fenics_function).__name__ == 'Function':
			pass
		else:
			fenics_function_array = fenics_function

		# Flag to indicate whether to apply changes or not
		set_to_apply_changes = False

		# OpenFOAM uses a 'rho-normalized pressure' (Pa/(kg/m³)) for incompressible flow. This means that,
		 # if we are using the "usual" pressure (Pa), we have to multiply/divide the FEniCS value 
		 # by the density.
		if foam_variable_name == 'p' and convert_to_usual_units_if_necessary == True:

			if self.dictionary_measurement_units['p'] == 'rho-normalized pressure':

				if 'fenics_function_array' not in locals():
					fenics_function_array = fenics_function.vector().get_local()

				if type_of_post_processing == 'from Foam':
					fenics_function_array *= self.additional_properties['rho']
				elif type_of_post_processing == 'to Foam':
					fenics_function_array /= self.additional_properties['rho']
				else:
					raise ValueError(" ❌ ERROR: type_of_post_processing == '%s' is not defined!" %(type_of_post_processing))

				# Set to apply changes
				set_to_apply_changes = True

		if foam_variable_name in [

			# Turbulent variables
			'k', 'epsilon', 'omega', 'nuTilda', 'v2', 'f', 

			# Turbulent kinematic viscosity
			'nut',

			# Turbulent thermal diffusivity
			'alphat',

			]:

			# Array of values
			if 'fenics_function_array' not in locals():
				fenics_function_array = fenics_function.vector().get_local()

			if type(self.min_value_projected_turbulence_variables).__name__ != 'NoneType': 

				# Impose that they should be always positive (i.e., ignoring possible numerical errors of the projection)
				fenics_function_array[fenics_function_array < 0] = self.min_value_projected_turbulence_variables #0

				# Set to apply changes
				set_to_apply_changes = True

		elif foam_variable_name == 'alpha_design':

			# Array of values
			if 'fenics_function_array' not in locals():
				fenics_function_array = fenics_function.vector().get_local()

			# Impose that they should be always in [1, 0] (i.e., ignoring possible numerical errors of the projection)
			fenics_function_array[fenics_function_array < 0] = 0
			fenics_function_array[fenics_function_array > 1] = 1

			# Set to apply changes
			set_to_apply_changes = True

		else:
			pass

		# Set to the FEniCS variable
		if type(fenics_function).__name__ == 'Function':
			if set_to_apply_changes == True:
				fenics_function.vector().set_local(fenics_function_array)
				fenics_function.vector().apply('insert')
		else:
			return fenics_function_array

	################## setAllEqualFoamBoundaryConditions ###################

	@utils_fenics.no_annotations()
	def setAllEqualFoamBoundaryConditions(self, variable_name, bc_type, bc_value, other_parameters = {}, check_consistency_of_imposed_values = False, convert_from_usual_units_if_necessary = True):
		"""
		Set all boundary conditions from FEniCS variable (or a string),
		except 'frontSurface', 'backSurface', 'frontAndBackSurfaces',
		'cyclic', 'cyclicAMI', 'cyclicACMI'.
		"""

		# Boundary data
		mesh_boundary_data = self.foam_solver.foam_mesh.boundaries()

		# Boundary names
		boundary_names = list(mesh_boundary_data.keys())

		for boundary_name in boundary_names:
			if boundary_name in ['frontSurface', 'backSurface', 'frontAndBackSurfaces']:
				pass
			else:

				if mesh_boundary_data[boundary_name]['type'] in ['cyclic', 'cyclicAMI', 'cyclicACMI']:
					self.setFoamBoundaryCondition(variable_name, boundary_name, bc_type, bc_value, other_parameters = other_parameters, check_consistency_of_imposed_values = check_consistency_of_imposed_values, return_error_if_boundary_unavailable = False, convert_from_usual_units_if_necessary = convert_from_usual_units_if_necessary, set_only_the_value = True)
				else:
					self.setFoamBoundaryCondition(variable_name, boundary_name, bc_type, bc_value, other_parameters = other_parameters, check_consistency_of_imposed_values = check_consistency_of_imposed_values, return_error_if_boundary_unavailable = False, convert_from_usual_units_if_necessary = convert_from_usual_units_if_necessary)

	####################### setFoamBoundaryCondition #######################

	@utils_fenics.no_annotations()
	def setFoamBoundaryCondition(self, 
		variable_name, 
		boundary_name, bc_type, 
		bc_value, 
		other_parameters = {}, 

		check_consistency_of_imposed_values = False, 
		return_error_if_boundary_unavailable = True, 
		convert_from_usual_units_if_necessary = True, 
		set_only_the_value = False
		):
		"""
		Set a boundary condition from FEniCS variable (or a string).
		"""

		# Get the FoamMesh
		foam_mesh = self.foam_solver.foam_mesh

		# Get the faces of the boundary
		boundary = foam_mesh.boundary(boundary_name)
		nFaces = boundary['nFaces']
		startFace = boundary['startFace']

		# Checking if the face has any faces marked with the given boundary
		if nFaces == 0:
			if return_error_if_boundary_unavailable == False:
				utils.customPrint(" ❗ There are no faces available for boundary '%s'!" %(boundary_name))
				return
			else:
				raise ValueError(" ❌ ERROR: There are no faces available for boundary '%s'!" %(boundary_name))

		utils.customPrint("\n 🌊 Setting boundary condition '%s' of variable '%s' to OpenFOAM:" %(boundary_name, variable_name))
		utils.customPrint(" - type = %s" %(bc_type))
		utils.customPrint(" - value = %s" %(bc_value))
		if len(other_parameters) == 0:
			utils.customPrint(" - other_parameters = [empty]")
		else:
			utils.customPrint(" - other_parameters = ")
			utils.printFullDictionary(other_parameters)

		def check_for_user_expression(fenics_var):
			"""
			Since UserExpressions are created by defining
			classes in Python, the type(...).__name__ scheme
			should not work. This is a workaround.
			"""

			try:
				if 'expression' in str(fenics_var.cpp_object()):
					is_expression = True
				else:
					is_expression = False
			except:
				is_expression = False

			return is_expression

		def get_np_array_of_bc(bc_value, foam_vector):
			"""
			Get the NumPy array of the boundary condition.
			"""
			if type(bc_value).__name__ == 'NoneType':
				value = None

			elif type(bc_value).__name__ in ['int', 'float']:
				value = np.array([bc_value])

			elif type(bc_value).__name__ == 'list':
				value = np.array(bc_value)

			elif type(bc_value).__name__ == 'ndarray':
				value = bc_value

			elif type(bc_value).__name__ == 'Constant':
				value = bc_value.values()

			elif type(bc_value).__name__ == 'Function' or type(bc_value).__name__ == 'Expression' or type(bc_value).__name__ == 'CompiledExpression' or check_for_user_expression(bc_value):

				# FEniCS variable
				fenics_var = bc_value

				# Compute the central coordinate of each face in 2D
				faces_central_coords = foam_mesh.faces_center_coordinates(boundary_name = boundary_name)

				# Convert to 2D coordinates if necessary
				faces_central_coords_adjusted = utils_fenics.adjustCoordinatesOpenFOAMToFEniCS(faces_central_coords, self.domain_type)

				# Set all central values from the FEniCS variable
				value = np.array([fenics_var(face_central_coords) for face_central_coords in faces_central_coords_adjusted], dtype = 'float')

				# Checking the value
				if len(value) == 0:
					raise ValueError(" ❌ ERROR: Imposing nothing on boundary '%s' for variable '%s'!" %(boundary_name, variable_name))

				if len(value.shape) == 1:
					pass
				elif len(value.shape) == 2:
					# Adjust the vector components according to the domain type
					value = utils_fenics.adjustVectorComponentsToOpenFOAM(value, self.domain_type)
				else:
					raise ValueError(" ❌ ERROR: len(value.shape) == %s is not valid!" %(len(value.shape)))

			else:
				value = bc_value

			if convert_from_usual_units_if_necessary == True and type(bc_value).__name__ != 'NoneType':
				value = self.postProcessFEniCSVariableToFoam(variable_name, value, convert_to_usual_units_if_necessary = convert_from_usual_units_if_necessary)

			# Check consistency
			if check_consistency_of_imposed_values == True:

				# Check if no 'nan' value appeared
				assert np.any(np.isnan(value)) == False

				# Check if no 'inf' value appeared
				assert np.any(np.isinf(value)) == False

			return value

		foam_vectors = self.foam_solver.getFoamVectors()
		foam_vector_names = {foam_vectors[i_foam_vector].name : i_foam_vector for i_foam_vector in range(len(foam_vectors))}

		if variable_name in foam_vector_names:

			# FoamVector
			foam_vector = foam_vectors[foam_vector_names[variable_name]]

			# Get the value
			value = get_np_array_of_bc(bc_value, foam_vector)

			# Check if there are values in other_parameters that need to be converted to NumPy arrays
			other_parameters = other_parameters.copy() # Let's make a copy

			# Run the recursive check for other_parameters
			def recursive_check(dictionary):
				for key in dictionary:
					if type(dictionary[key]).__name__ in ['str', 'int', 'float', 'list', 'NoneType']: # Something to set as is
						pass
					elif type(dictionary[key]).__name__ == 'dict':
						recursive_check(dictionary[key]) # Recursion
					else:
						dictionary[key] = get_np_array_of_bc(dictionary[key], foam_vector) # Array to set
			recursive_check(other_parameters)

			# Set the boundary condition
			if set_only_the_value == True:
				foam_vector.setBoundaryConditionValue(boundary_name, value)
			else:
				foam_vector.setBoundaryCondition(boundary_name, bc_type, value, other_parameters = other_parameters)

			# Set to apply changes
			foam_vector.set_to_apply_changes('insert')

		else:
			raise ValueError(" ❌ ERROR: variable_name == '%s' not in %s!" %(variable_name, foam_vector_names))

	######################### setFoamConfiguration #########################

	@utils_fenics.no_annotations()
	def setFoamConfiguration(self, configuration_name, new_data):
		"""
		Set an OpenFOAM configuration.
		"""

		if len(new_data) != 0 and type(new_data).__name__ != 'NoneType':

			utils.customPrint("\n 🌊 Setting configuration to OpenFOAM: %s" %(configuration_name))
			utils.printFullDictionary(new_data)

			foam_configurations = self.foam_solver.getFoamConfigurations()
			foam_configurations_names = {foam_configurations[i_foam_configuration].name : i_foam_configuration for i_foam_configuration in range(len(foam_configurations))}

			if configuration_name in foam_configurations_names:
				foam_configuration = foam_configurations[foam_configurations_names[configuration_name]]
				foam_configuration.setConfiguration(new_data)
				foam_configuration.set_to_apply_changes('insert')
			else:
				raise ValueError(" ❌ ERROR: configuration_name == '%s' not in %s!" %(configuration_name, foam_configurations_names))

	########################### setFoamProperty ############################

	@utils_fenics.no_annotations()
	def setFoamProperty(self, property_name, new_data):
		"""
		Set an OpenFOAM property.
		"""

		if len(new_data) != 0 and type(new_data).__name__ != 'NoneType':

			utils.customPrint("\n 🌊 Setting property to OpenFOAM: %s" %(property_name))
			utils.printFullDictionary(new_data)

			foam_properties = self.foam_solver.getFoamProperties()
			foam_properties_names = {foam_properties[i_foam_property].name : i_foam_property for i_foam_property in range(len(foam_properties))}

			if property_name in foam_properties_names:
				foam_property = foam_properties[foam_properties_names[property_name]]
				foam_property.setProperty(new_data)
				foam_property.set_to_apply_changes('insert')
			else:
				raise ValueError(" ❌ ERROR: property_name == '%s' not in the recognized property dictionaries (%s)! Remember that all property dictionary names MUST end with 'Properties'!" %(property_name, foam_properties_names))

	####################### setAdditionalProperty ##########################

	@utils_fenics.no_annotations()
	def setAdditionalProperty(self, name, value):
		"""
		Set some additional properties that may be needed here.
		* For example, one is "rho" (density), because it is needed
		  to convert the measurement unit of pressure used by OpenFOAM
		  ("rho-normalized pressure") (Pa/(kg/m³)) to the usual
		  unit of pressure (Pa) (in incompressible solvers).
		"""

		if type(value).__name__ == 'Constant':
			self.additional_properties[name] = float(value)
		elif type(value).__name__ in ['float', 'int']:
			self.additional_properties[name] = value
		else:
			raise ValueError(" ❌ ERROR: type(value).__name__ == '%s' is not defined!" %( type(value).__name__))

	############################# plotResults ##############################

	@utils_fenics.no_annotations()
	def plotResults(self, file_type = 'VTK', tag_folder_name = "", more_options_for_export = ""):
		"""
		Plot the results to files.
		"""

		self.foam_solver.plotResults(file_type = file_type, tag_folder_name = tag_folder_name, more_options_for_export = more_options_for_export)

	################################ solve #################################

	@utils_fenics.no_annotations()
	def solve(self, 
		run_mode = 'openfoam', 
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
		Solve the problem with FoamSolver.
		"""

		# Let's overload the 'error_on_nonconvergence' parameter from FoamSolver
		error_on_nonconvergence_BAK = self.foam_solver.parameters['error_on_nonconvergence']
		self.foam_solver.parameters['error_on_nonconvergence'] = False

		# Solve with FoamSolver
		try:
			result_info = self.foam_solver.solve(
				run_mode = run_mode, 
				consider_that_we_may_have_successive_simulation_parameters = consider_that_we_may_have_successive_simulation_parameters,

				# Log file
				save_log_file = save_log_file, 
				log_file = log_file,
				only_print_to_log_file = only_print_to_log_file,

				# Silent mode
				silent_run_mode = silent_run_mode,
				num_logfile_lines_to_print_in_silent_mode = num_logfile_lines_to_print_in_silent_mode,

				# MPI configs
				mpi_configs = mpi_configs,

				# Continuous plotting of residuals
				continuously_plot_residuals_from_log = continuously_plot_residuals_from_log,
				continuously_plot_residuals_from_log_tag = continuously_plot_residuals_from_log_tag,
				continuously_plot_residuals_from_log_time_interval = continuously_plot_residuals_from_log_time_interval,
				continuously_plot_residuals_from_log_x_axis_label = continuously_plot_residuals_from_log_x_axis_label,
				continuously_plot_residuals_from_log_y_axis_scale = continuously_plot_residuals_from_log_y_axis_scale,
				continuously_plot_residuals_from_log_use_lowest_priority_for_plotting = continuously_plot_residuals_from_log_use_lowest_priority_for_plotting,
				)

			# Successful
			solver_worked = True

			# Set the last step as the result
			time_step_name = 'last'

		except:
			import traceback
			traceback.print_exc()

			# Diverged!
			result_info = [0, False]

			# Not successful
			solver_worked = False

			# Set the initial guess as the result
			time_step_name = 0

		# Prepare for editing the last step
		 # * It IS necessary to always run the line below, even when the 
		 #   OpenFOAM simulation fails.
		 #   Why? Because the user may want to use a "try-except" clause
		 #   around "fenics_foam_solver.solve". If that happens, we need 
		 #   to get everything ready for the user to edit.
		self.foam_solver.prepareForEditing(time_step_name = time_step_name, keep_previous_setup = True, keep_previous_mesh = True)

		# Renew the foam_vectors dictionary
		foam_vectors = self.foam_solver.getFoamVectors()
		foam_vector_names = {foam_vectors[i_foam_vector].name : i_foam_vector for i_foam_vector in range(len(foam_vectors))}

		for variable_name in foam_vector_names:
			if variable_name in self.variable_dictionary:
				self.variable_dictionary[variable_name]['FoamVector'] = foam_vectors[foam_vector_names[variable_name]]

		# Now let's return the previous 'error_on_nonconvergence' parameter from FoamSolver
		self.foam_solver.parameters['error_on_nonconvergence'] = error_on_nonconvergence_BAK

		# Check convergence
		converged = result_info[1]
		if converged == False:
			if self.foam_solver.parameters['error_on_nonconvergence'] == True:
				last_time_step = result_info[0]
				raise ValueError(" ❌ ERROR: OpenFOAM solver did not converge until time step %s!" %(last_time_step))

		# Solver failed?
		if solver_worked == False:
			if silent_run_mode == False:
				raise ValueError(" ❌ ERROR: Some problem occurred in the OpenFOAM simulation!")
			else:
				if save_log_file == True:
					raise ValueError(" ❌ ERROR: Some problem occurred in the OpenFOAM simulation! Check the generated 'foam_log' to see what happened in OpenFOAM!")
				else:
					raise ValueError(" ❌ ERROR: Some problem occurred in the OpenFOAM simulation! Who knows what happened! Please, enable 'save_log_file = True' or use 'silent_run_mode = False' in order to try to discover what happened!")

		return result_info

############################## Load plugins ####################################

from ..plugins import load_plugins
load_plugins.loadPlugins(__file__, globals())

################################################################################

################################################################################
#                                 utils_fenics                                 #
################################################################################
# Some utilities for using with FEniCS.

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

############################### FEniCS libraries ###############################

# FEniCS
try:
	import fenics
	from fenics import *
	 # Do not import dolfin-adjoint here. This is a non-annotated module!
except:
	utils.printDebug(" ❌ FEniCS not installed! Can't use FEniCSFoamSolver!")

################################# no_annotations ###############################

def no_annotations(*args, **kwargs):
	"""
	Decorator to avoid dolfin-adjoint/pyadjoint annotations.

	-> Example of usage:
		@utils_fenics.no_annotations()
		def f(*args, **kwargs):
			[...]
	"""

	try: # If pyadjoint is installed, let's guarantee that no annotations will be performed!

		import pyadjoint
		from pyadjoint.tape import no_annotations
		decorator = no_annotations

	except: # pyadjoint is not available in your installation, so do nothing here!

		def decorator(original_function):
			"""
			Dummy decorator. Does nothing at all.
			"""
			def wrapper(*args, **kwargs):
				return original_function(*args, **kwargs)
			return wrapper

	return decorator

############################### configuredProject ##############################

def configuredProject(variable, function_space_for_projection, projection_setup, skip_projection_if_same_FunctionSpace = False):
	"""
	Interfacing the FEniCS project function in order to allow the 
	user to set advanced options of the project function.

	* One possible use is changing the 'quadrature_degree' option in order
	  to try improving the numerical accuracy of the projection.

	** This function has been crafted so as not to follow the 'quadrature_degree'
	   set in the FEniCS 'parameter' dictionary, which means that you can change
	   the quadrature degree of this function independently from your global
	   setup.

	"""

	#### Copy the projection setup
	projection_setup = projection_setup.copy()

	#### If trying to skip projection
	if skip_projection_if_same_FunctionSpace == True:

		# If the variable is a Function
		if type(variable).__name__ == 'Function':

			# If both FunctionSpace's are the same
			if variable.function_space().__repr__() == function_space_for_projection.__repr__():

				# New variable with the same values
				new_variable = Function(function_space_for_projection)
				new_variable.assign(variable)

				#variable_array = new_variable.vector().get_local()
				#new_variable.vector().set_local(variable_array)
				#new_variable.apply('insert')

				return new_variable

	#### solver_type
	solver_type = projection_setup.get('solver_type', 'lu')

	#### preconditioner_type
	preconditioner_type = projection_setup.get('preconditioner_type', 'default')

	#### Form compiler parameters
	form_compiler_parameters = projection_setup.get('form_compiler_parameters', {}).copy()

	#  Quadrature degree
	assert 'quadrature_degree' not in form_compiler_parameters
	type_of_quadrature_degree = form_compiler_parameters.pop('type_of_quadrature_degree', 'from FEniCS parameters dictionary')

	if type_of_quadrature_degree == 'auto':
		form_compiler_parameters['quadrature_degree'] = None # (* None = Use the automatic quadrature degree)
	elif type_of_quadrature_degree == 'from FEniCS parameters dictionary':
		pass
	elif 'int' in type(type_of_quadrature_degree).__name__:
		form_compiler_parameters['quadrature_degree'] = type_of_quadrature_degree
	else:
		raise ValueError(" ❌ ERROR: type_of_quadrature_degree == '%s' is not defined!" %(type_of_quadrature_degree))

	# Check max quadrature degree
	max_quadrature_degree = projection_setup.pop('max_quadrature_degree', None)
	if (type(max_quadrature_degree).__name__ == 'NoneType') or ('int' in type(type_of_quadrature_degree).__name__):
		pass
	else:

		try:

			form = inner(variable, TestFunction(function_space_for_projection))*dx(function_space_for_projection.mesh())
			quadrature_rule, quadrature_degree = checkAutomaticQuadratureDegree(form)

			if quadrature_degree > max_quadrature_degree:

				utils.customPrint(" ❗ Automatic quadrature degree determined for projection: %d -> Reduced to %d" %(quadrature_degree, max_quadrature_degree))

				form_compiler_parameters['quadrature_degree'] = max_quadrature_degree

			else:
				utils.customPrint(" ❗ Automatic quadrature degree determined for projection: %d" %(quadrature_degree))

		except:
			import traceback
			traceback.print_exc()
			utils.customPrint(" ❗ Unable to automatically determine the quadrature degree for projection!")

	# Project
	return project(variable, function_space_for_projection, 
		solver_type = solver_type,
		preconditioner_type = preconditioner_type, 
		form_compiler_parameters = form_compiler_parameters
		)

####################### checkAutomaticQuadratureDegree #########################

def checkAutomaticQuadratureDegree(form):
	"""
	Check the automatically determined quadrature degree by FFC (ffc.analysis).

	Code HIGHLY based on the "ffc.analysis" module in FEniCS 2019.1.0.

	Copyright disclaimer of the original code that served as the basis for this function:
		# Copyright (C) 2007-2017 Anders Logg, Martin Alnaes, Kristian B. Oelgaard,
		# and others
		#
		# This file is part of FFC.
		#
		# FFC is free software: you can redistribute it and/or modify
		# it under the terms of the GNU Lesser General Public License as published by
		# the Free Software Foundation, either version 3 of the License, or
		# (at your option) any later version.
		#
		# FFC is distributed in the hope that it will be useful,
		# but WITHOUT ANY WARRANTY; without even the implied warranty of
		# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
		# GNU Lesser General Public License for more details.
		#
		# You should have received a copy of the GNU Lesser General Public License
		# along with FFC. If not, see <http://www.gnu.org/licenses/>.

	"""

	import ffc
	import copy

	parameters_dict = ffc.parameters.default_parameters()

	# Parameter values which make sense "per integrals" or "per integral"
	metadata_keys = (
		"representation",
		"optimize",
		"precision",
		"quadrature_degree",
		"quadrature_rule",
	)

	# Get defaults from parameters
	metadata_parameters = {key: parameters_dict[key] for key in metadata_keys if key in parameters_dict}

	# Representation of the integral
	if "representation" in metadata_parameters:
		assert metadata_parameters['representation'] in ['quadrature', 'auto']
	form_representation_family = 'quadrature'

	# Compute form data for 'quadrature' representation
	form_data = ffc.analysis.compute_form_data(form)

	# Iterate over integral collections
	quad_schemes = []
	for ida in form_data.integral_data:

		# Iterate over integrals
		# Start with default values of integral metadata
		# (these will be either the FFC defaults, globally modified defaults,
		#  or overrides explicitly passed by the user to e.g. assemble())
		integral_metadatas = [copy.deepcopy(metadata_parameters) for integral in ida.integrals]

		# Update with integral specific overrides
		for i, integral in enumerate(ida.integrals):

			integral_metadatas[i].update(integral.metadata() or {})

			# Determine representation, must be equal for all integrals on
			# same subdomain

			r, o, p = ffc.analysis._determine_representation(integral_metadatas, ida, form_data, form_representation_family, parameters_dict)
			for i, integral in enumerate(ida.integrals):
				integral_metadatas[i]["representation"] = r
				integral_metadatas[i]["optimize"] = o
				integral_metadatas[i]["precision"] = p

			ida.metadata["representation"] = r
			ida.metadata["optimize"] = o
			ida.metadata["precision"] = p

			# Determine automated updates to metadata values
			for i, integral in enumerate(ida.integrals):
				quadrature_rule = ffc.analysis._autoselect_quadrature_rule(integral_metadatas[i], integral, form_data)
				quadrature_degree = ffc.analysis._autoselect_quadrature_degree(integral_metadatas[i], integral, form_data)
				integral_metadatas[i]["quadrature_rule"] = quadrature_rule
				integral_metadatas[i]["quadrature_degree"] = quadrature_degree

		# Extract common metadata for integral collection
		quadrature_rule = ffc.analysis._extract_common_quadrature_rule(integral_metadatas)
		quadrature_degree = ffc.analysis._extract_common_quadrature_degree(integral_metadatas)

	return quadrature_rule, quadrature_degree

################################################################################
###################### FEniCS <-> OpenFOAM internalField #######################
################################################################################

######################### createDG0FEniCSFunctionSpace #########################

def createDG0FEniCSFunctionSpace(mesh, dim = 1):
	"""
	Create DG0 FEniCS FunctionSpace.
	"""

	# Create a DG0 function space
	if dim == 0:
		function_space = FunctionSpace(mesh, 'DG', 0)
	else:
		function_space = VectorFunctionSpace(mesh, 'DG', 0, dim = dim)

	return function_space

############################ createDG0FEniCSFunction ###########################

def createDG0FEniCSFunction(mesh, dim = 1):
	"""
	Create DG0 FEniCS Function.
	"""

	# Create a DG0 function space
	function_space = createDG0FEniCSFunctionSpace(mesh, dim = dim)

	# Create the FEniCS Function
	fenics_function = Function(function_space)

	return fenics_function

########################## FoamVectorToFEniCSFunction ##########################

def FoamVectorToFEniCSFunction(foam_vector, fenics_function, map_DoFs_fenics_to_foam, maps_component_function, domain_type):
	"""
	Set the values of a FoamVector to a Function in FEniCS.
	"""

	# Get the vector from the FEniCS Function
	fenics_vector = fenics_function.vector()

	# Get the FEniCS DoF (Degrees of Freedom) coordinates
	#fenics_DoF_coordinates = getDoFCoordinatesFromFunctionSpace(fenics_function.function_space())

	# Get the array from the FEniCS Function
	fenics_function_array = fenics_function.vector().get_local()

	# Get the array from FoamVector
	foam_vector_array = foam_vector.get_local()

	# Set the new values for the FEniCS Function
	if foam_vector.num_components == 1:
		fenics_function_array = np.array([foam_vector.value_from_index(map_DoFs_fenics_to_foam[i_fenics]) for i_fenics in range(len(fenics_function_array))])

	elif foam_vector.num_components == 3:
		if fenics_function.function_space().num_sub_spaces() == 3:
			map_from_function_to_component = maps_component_function['3 components']['from function to component']
		else:
			map_from_function_to_component = maps_component_function['2 components']['from function to component']

		def mapFromComponents(i_fenics):
			"""
			Maps the component of the FoamVector that corresponds to 
			i_fenics to the component of the FEniCS Function.
			When the FEniCS Function has 2 components, one component of the 
			FoamVector is ignored. 
			"""
			[num_component, i_fenics_component] = map_from_function_to_component[i_fenics]
			i_foam_component = map_DoFs_fenics_to_foam[i_fenics_component]
			return foam_vector_array[i_foam_component][num_component]

		fenics_function_array = np.array([mapFromComponents(i_fenics) for i_fenics in range(len(fenics_function_array))])

	else:
		raise ValueError(" ❌ ERROR: foam_vector.num_components == %d is not implemented!" %(foam_vector.num_components))

	# Set the array for FEniCS
	fenics_vector.set_local(fenics_function_array)
	fenics_vector.apply('insert')

######################### FEniCSFunctionToFoamVector ###########################

def FEniCSFunctionToFoamVector(fenics_function, foam_vector, map_DoFs_foam_to_fenics, maps_component_function, domain_type):
	"""
	Set the values of a variable in FEniCS to a FoamVector.
	"""

	# Get the vector from the FEniCS Function
	fenics_vector = fenics_function.vector()

	# Get the FEniCS DoF (Degrees of Freedom) coordinates
	#fenics_DoF_coordinates = getDoFCoordinatesFromFunctionSpace(fenics_function.function_space())

	# Get the array from the FEniCS Function
	fenics_function_array = fenics_vector.get_local()

	# Get the array from FoamVector
	foam_vector_array = foam_vector.get_local()

	# Set the new values for the FoamVector
	if foam_vector.num_components == 1:
		foam_vector_array = np.array([fenics_function_array[map_DoFs_foam_to_fenics[i_foam]] for i_foam in range(len(foam_vector_array))])
	elif foam_vector.num_components == 3:
		if fenics_function.function_space().num_sub_spaces() == 3:
			map_from_component_to_function = maps_component_function['3 components']['from component to function']

			def mapTo3components(i_foam):
				"""
				Maps the component of the FEniCS Function that corresponds to 
				i_foam to the component of the FoamVector.
				"""
				i_fenics = map_DoFs_foam_to_fenics[i_foam]
				return [fenics_function_array[map_from_component_to_function[i_component][i_fenics]] for i_component in range(3)]

		else:
			map_from_component_to_function = maps_component_function['2 components']['from component to function']

			def mapTo3components(i_foam):
				"""
				Maps the component of the FEniCS Function that corresponds to 
				i_foam to the component of the FoamVector.
				In this case, the FEniCS Function has 2 components, one component of the 
				FoamVector is set as 0.0. 
				"""
				i_fenics = map_DoFs_foam_to_fenics[i_foam]

				if domain_type == '2D': # Ignore third component

					return [
						fenics_function_array[map_from_component_to_function[0][i_fenics]],	# v_x
						fenics_function_array[map_from_component_to_function[1][i_fenics]],	# v_y
						0.0									# v_z
						]

				elif domain_type == '2D axisymmetric': # If you are (for some unknown reason) using only v_r and v_z (without v_θ)

					return [
						fenics_function_array[map_from_component_to_function[0][i_fenics]], 	# v_r
						0.0,								  	# v_θ 
						fenics_function_array[map_from_component_to_function[1][i_fenics]],	# v_z
						]
				else:
					raise ValueError(" ❌ ERROR: domain_type == %s is not defined here!" %(domain_type))

		foam_vector_array = np.array([mapTo3components(i_foam) for i_foam in range(len(foam_vector_array))])
	else:
		raise ValueError(" ❌ ERROR: foam_vector.num_components == %d is not implemented!" %(foam_vector.num_components))

	# Set the array for FoamVector
	foam_vector.set_local(foam_vector_array)
	foam_vector.set_to_apply_changes('insert')

################################ floatVector ###################################

def floatVector(fenics_constant):
	"""
	Extends the "float" in FEniCS for vectors of the type "Constant".
	-> Converts a Constant vector into an array of floats
	-> https://fenicsproject.org/qa/592/how-can-one-get-the-value-of-a-vector-constant/
	"""

	var = fenics_constant
	var_shape = var.ufl_shape
	var_num_dims = len(var_shape)

	if var_num_dims == 0:
		vals = float(var)
	elif var_num_dims == 1:
		vals = np.zeros(var_shape)
		var.eval(vals, np.zeros(var_shape))
	else:
		raise ValueError(" ❌ ERROR: floatVector not implemented for dim = %d!" % (var_num_dims))

	return vals

##################### adjustCoordinatesOpenFOAMToFEniCS ########################

def adjustCoordinatesOpenFOAMToFEniCS(foam_coordinates, domain_type):
	"""
	Adjust the coordinates to FEniCS.
	By using advanced slicing.
		https://stackoverflow.com/questions/4857927/swapping-columns-in-a-numpy-array
	"""

	if domain_type == '2D':
		# (x, y, z) ---> (x, y)
		return foam_coordinates[:,[0, 1]] # Remove third component of the coordinates of the mesh in OpenFOAM
			# foam_coordinates[:,:-1]
	elif domain_type == '2D axisymmetric':
		# (r, θ, z) ---> (r, z)
		return foam_coordinates[:,[0, 2]] # Remove the middle component of the coordinates of the mesh in OpenFOAM
	elif domain_type == '3D':
		return foam_coordinates # Return the coordinates of the mesh in OpenFOAM
	else:
		raise ValueError(" ❌ ERROR: domain_type == '%s' is not defined!" %(domain_type))

##################### adjustCoordinatesFEniCSToOpenFOAM ########################

def adjustCoordinatesFEniCSToOpenFOAM(fenics_coordinates, domain_type):
	"""
	Adjust the coordinates to OpenFOAM.
	By using the NumPy trick for translation of slices.
		https://docs.scipy.org/doc/numpy/reference/generated/numpy.c_.html
		https://stackoverflow.com/questions/8486294/how-to-add-an-extra-column-to-a-numpy-array#
	"""

	if domain_type == '2D':
		# (x, y) ---> (x, y, z)
		assert len(fenics_coordinates[0]) == 2
		return np.c_[fenics_coordinates[:,0], fenics_coordinates[:,1], np.zeros(len(fenics_coordinates))] # Include a zero third component for the coordinates of the mesh in OpenFOAM
	elif domain_type == '2D axisymmetric':
		# (r, θ, z) ---> (r, z)
		assert len(fenics_coordinates[0]) == 3
		return np.c_[fenics_coordinates[:,0], np.zeros(len(fenics_coordinates)), fenics_coordinates[:,1]] # Include the middle component of the coordinates for the mesh in OpenFOAM

	elif domain_type == '3D':
		return fenics_coordinates # Return the coordinates of the mesh in OpenFOAM
	else:
		raise ValueError(" ❌ ERROR: domain_type == '%s' is not defined!" %(domain_type))

###################### adjustVectorComponentsToOpenFOAM ########################

def adjustVectorComponentsToOpenFOAM(vector_components_array, domain_type):
	"""
	Adjust the vector components to OpenFOAM.
	By using the NumPy trick for translation of slices.
		https://docs.scipy.org/doc/numpy/reference/generated/numpy.c_.html
		https://stackoverflow.com/questions/8486294/how-to-add-an-extra-column-to-a-numpy-array#
	"""

	if domain_type == '2D':
		# (v_x, v_y) ---> (v_x, v_y, v_z)
		assert len(vector_components_array[0]) == 2
		return np.c_[vector_components_array[:,0], vector_components_array[:,1], np.zeros(len(vector_components_array))] # Include a zero third component for the vector components for OpenFOAM
	elif domain_type == '2D axisymmetric' or domain_type == '3D':
		# (v_r, v_θ, v_z)
		# (v_x, v_y, v_z)
		assert len(vector_components_array[0]) == 3
		return vector_components_array # Return the vector components for OpenFOAM
	else:
		raise ValueError(" ❌ ERROR: domain_type == '%s' is not defined!" %(domain_type))

####################### generateDoFMapsOpenFOAM_FEniCS #########################

def generateDoFMapsOpenFOAM_FEniCS(
		fenics_dof_coordinates, 
		foam_dof_coordinates, 
		domain_type, 
		mesh_hmin = 1.E-6, # Minimum size of element in the mesh
		tol_factor = 0.005
		):
	"""
	Generates the maps between OpenFOAM and FEniCS DoFs.
	* DoFs = Degrees of Freedom
	"""

	# Adjust coordinates in case of 2D mesh
	foam_coords_to_search = adjustCoordinatesOpenFOAMToFEniCS(foam_dof_coordinates, domain_type)

	# Create the maps between coordinate arrays
	#tol_mapping = 0.05*mesh_hmin # 5% of tolerance
	#tol_mapping = 0.01*mesh_hmin # 1% of tolerance
	#tol_mapping = 0.005*mesh_hmin # 0.5% of tolerance
	tol_mapping = tol_factor*mesh_hmin # 0.5% of tolerance
	(map_DoFs_fenics_to_foam, map_DoFs_foam_to_fenics) = utils.createMapsBetweenCoordinateArrays(fenics_dof_coordinates, foam_coords_to_search, tol_mapping = tol_mapping)

	#### Previous version -- Surely works, but it can become quite slow in bigger meshes

#	# Print progress
#	utils.printProgress(0, max_iterations = len(foam_dof_coordinates))

#	map_DoFs_foam_to_fenics = {}
#	map_DoFs_fenics_to_foam = {}

#	# Go through all vertices of the OpenFOAM mesh and find the equivalent in the FEniCS mesh
#	for i_foam in range(len(foam_dof_coordinates)):
#	
#		foam_coords = foam_coords_to_search[i_foam]

#		i_fenics = utils.findNearestSubArray(fenics_dof_coordinates, foam_coords, parameters_to_return = 'index')

#		map_DoFs_foam_to_fenics[i_foam] = i_fenics
#		map_DoFs_fenics_to_foam[i_fenics] = i_foam

#		# Print progress
#		utils.printProgress(i_foam + 1)

	return map_DoFs_foam_to_fenics, map_DoFs_fenics_to_foam

#################### getDoFCoordinatesFromFunctionSpace ########################

def getDoFCoordinatesFromFunctionSpace(U):
	"""
	Returns the coordinates of the DoFs.
	* DoFs = Degrees of Freedom
	https://fenicsproject.org/qa/10782/tabulate_all_coordinates-removed-in-2016-1-0
	https://fenicsproject.org/qa/2715/coordinates-u_nodal_values-using-numerical-source-function
	"""
	#mesh = U.mesh()
	#dim = U.dim() # * This dimension is the quantity of values in a function of the function space
	#N = mesh.geometry().dim()
	#DoF_mesh_coordinates = U.tabulate_dof_coordinates().reshape(dim, N)

	DoF_mesh_coordinates = U.tabulate_dof_coordinates()

	return DoF_mesh_coordinates

################ getDoFmapFromFEniCSVectorFunctiontoComponent ##################

def getDoFmapFromFEniCSVectorFunctiontoComponent(U):
	"""
	Returns the map of DoFs from component to function.

	* DoF = Degree of Freedom
	"""

	# Extract subfunction DoFs (* Degrees of Freedom)
	num_subspaces = U.num_sub_spaces()
	if num_subspaces == 0:
		dofs_array = [U.dofmap().dofs()]
	else:
		dofs_array = [U.sub(i).dofmap().dofs() for i in range(num_subspaces)]

	def getCollapsedDoFs(U, num_subspace):
		(U_collapsed_i, collapsed_dofs_map_to_dof_i) = U.sub(num_subspace).collapse(collapsed_dofs = True)
		return collapsed_dofs_map_to_dof_i

	# Collapsed DoF's maps to DoF
	map_from_component_to_function = [getCollapsedDoFs(U, num_subspace) for num_subspace in range(len(dofs_array))]

	maps_from_function_to_component = [invertMapOrder(map_from_component_to_function[i]) for i in range(len(map_from_component_to_function))]

	def getCollapsedDoFfromFunction(i_function):

		found = False
		for num_subspace in range(len(dofs_array)):
			if i_function in maps_from_function_to_component[num_subspace]:
				component_number = num_subspace
				collapsed_dof_number = maps_from_function_to_component[num_subspace][i_function]
				found = True
				break

		if found == False:
			raise ValueError(" ❌ ERROR: i_function == %d is could not be found!" %(i_function))

		return [component_number, collapsed_dof_number]

	local_size = len(Function(U).vector().get_local())
		# In serial, using U.dim() is fine. In parallel, U.dim() is the global size...

	map_from_function_to_component = [getCollapsedDoFfromFunction(i_function) for i_function in range(local_size)]

	return map_from_component_to_function, map_from_function_to_component

############################### invertMapOrder #################################

def invertMapOrder(original_map, increment_new_key = 0):
	"""
	Inverts content and key of a dictionary.
	https://stackoverflow.com/questions/483666/reverse-invert-a-dictionary-mapping
	"""

	if type(original_map).__name__ == 'dict':
		if increment_new_key == 0:
			inverted_map = {v : k for k, v in original_map.items()}
		else:
			inverted_map = {v + increment_new_key : k for k, v in original_map.items()}
	elif type(original_map).__name__ in ['list', 'ndarray']:

		if type(original_map).__name__ == 'list':
			pass
		elif type(original_map).__name__ == 'ndarray':
			original_map = original_map.tolist()

		if increment_new_key == 0:
			inverted_map = {v : k for k, v in enumerate(original_map)}
		else:
			inverted_map = {v + increment_new_key : k for k, v in enumerate(original_map)}
	else:
		ValueError(" ❌ ERROR: type(original_map).__name__ == '%s' não está definido!" %(type(original_map).__name__))

	return inverted_map

####################### removeRepeatedElementsFromList #########################

def removeRepeatedElementsFromList(original_list, keep_order = True):
	"""
	Remove repeated elements from original_list.
	"""
	if keep_order == True:
		nova_list = sorted(set(original_list), key = original_list.index)
	else:
		nova_list = list(set(original_list))

	return nova_list

############################ mapBoundaryMeshToMesh #############################

def mapBoundaryMeshToMesh(bmesh, entity_type = 'vertex'):
	"""
	Map entities of a boundary mesh (BoundaryMesh) to the original mesh (Mesh).
	https://fenicsproject.discourse.group/t/confusion-on-mapping-boundarymesh-to-parent-mesh/1000
	"""

	if entity_type == 'vertex':
		num_ = 0

	elif entity_type == 'edge':
		num_ = 1

	elif entity_type == 'cell':
		num_ = bmesh.geometric_dimension()

	elif entity_type == 'facet':

		if bmesh.geometric_dimension() == 2: # 2D coordinates
			num_ = 1 # edge 
		elif bmesh.geometric_dimension() == 3: # 3D coordinates
			num_ = 2 # face
		else:
			raise ValueError(" ❌ ERROR: bmesh.geometric_dimension() == '%s' is not defined!" %(bmesh.geometric_dimension()))

	elif entity_type == 'face':
		if bmesh.geometric_dimension() == 2: # 2D coordinates
			raise ValueError(" ❌ ERROR: entity_type == '%s' is not defined for a 2D mesh!" %(entity_type))
		elif bmesh.geometric_dimension() == 3: # 3D coordinates
			num_ = 2
		else:
			raise ValueError(" ❌ ERROR: bmesh.geometric_dimension() == '%s' is not defined!" %(bmesh.geometric_dimension()))
	else:
		ValueError(" ❌ ERROR: entity_type == '%s' não está definido!" %(entity_type))

	try:
		map_from_bmesh_to_mesh = bmesh.entity_map(num_).array()
	except:
		raise ValueError(" ❌ ERROR: type(bmesh).__name__ == '%s' should be 'SubMesh' or an overload (from dolfin-adjoint, 'OverLoadedMesh') of SubMesh!" %(type(bmesh).__name__))

	return map_from_bmesh_to_mesh

############################# adjustTo3Dcomponents #############################

def adjustTo3Dcomponents(coords):
	"""
	Adjust to 3D components.
	"""

	if len(coords) == 2:
		return np.array([coords[0], coords[1], 0])
	elif len(coords) == 3:
		return coords
	else:
		raise ValueError(" ❌ ERROR: len(coords) == %d!" %(len(coords)))

############################## computeNormalVector #############################

def computeNormalVector(vertex_array, normalize = True):
	"""
	Compute normal vector to a face formed by a list of vertex_coordinates.
	"""

	# Compute the normal vector
	normal_vec = np.array([0, 0, 0], dtype = 'float')
	for i in range(len(vertex_array) - 1):
		if i == 0:
			ref_coordinate = adjustTo3Dcomponents(vertex_array[i])
		else:
			current_coordinate = adjustTo3Dcomponents(vertex_array[i])
			next_coordinate = adjustTo3Dcomponents(vertex_array[i+1])
			normal_vec += np.cross(current_coordinate - ref_coordinate, next_coordinate - ref_coordinate)

	# Normal vector
	if normalize == True:
		return normal_vec/np.linalg.norm(normal_vec)
	else: 
		return normal_vec # Not normalized => Magnitude of [Face area]/2

############################ copyListAndRemoveIndices ##########################

def copyListAndRemoveIndices(list_, indices_to_remove):
	"""
	Copy a list and remove indices.
	"""
	if type(indices_to_remove).__name__ != 'list':
		indices_to_remove = [indices_to_remove]

	# Indices to stay
	indices_to_stay = [i for i in range(len(list_))]
	[indices_to_stay.remove(indices_to_remove[i]) for i in range(len(indices_to_remove))]

	# new list with the indices to stay
	new_list = [list_[i_stay] for i_stay in indices_to_stay]

	return new_list

################################################################################
###################### FEniCS <-> OpenFOAM boundaryField #######################
################################################################################

######################### createCG1FEniCSFunctionSpace #########################

def createCG1FEniCSFunctionSpace(mesh, dim = 1):
	"""
	Create CG1 ('CG'/'Lagrange', degree = 1) FEniCS FunctionSpace.
	"""

	# Create a DG0 function space
	if dim == 0:
		function_space = FunctionSpace(mesh, 'Lagrange', 1)
	else:
		function_space = VectorFunctionSpace(mesh, 'Lagrange', 1, dim = dim)

	return function_space

############################ createCG1FEniCSFunction ###########################

def createCG1FEniCSFunction(mesh, dim = 1):
	"""
	Create CG1 ('CG'/'Lagrange', degree = 1) FEniCS Function.
	"""

	# Create a CG1 function space
	function_space = createCG1FEniCSFunctionSpace(mesh, dim = dim)

	# Create the FEniCS Function
	fenics_function = Function(function_space)

	return fenics_function

#################### generateFacetMapsOpenFOAM_FEniCSCells #####################

def generateFacetMapsOpenFOAM_FEniCSCells(fenics_mesh, foam_mesh, domain_type, mesh_hmin = 1.E-6, tol_factor = 0.005):
	"""
	Generates the maps between OpenFOAM and FEniCS facets.
	"""

	# FEniCS+MPI utilities
	from ..utils import utils_fenics_mpi

	# [FEniCS] Boundary mesh
	bmesh = BoundaryMesh(fenics_mesh, 'exterior', True)

	# [FEniCS in parallel] Unified mesh
	if utils_fenics_mpi.runningInParallel():

		# Wait for everyone!
		utils_fenics_mpi.waitProcessors()

		# Local/global meshes
		local_fenics_mesh = fenics_mesh
		global_foam_mesh = foam_mesh

		# Things to prepare beforehand, since they won't be accessible
		 # when computing the rest in a single processor.
		prepare_beforehand_mesh = [
			'geometric_dimension',
			'topology',
		]

		# Unify the mesh
		unified_fenics_mesh = utils_fenics_mpi.UnifiedMesh(local_fenics_mesh, 
			prepare_beforehand = prepare_beforehand_mesh,
			proc_destination = 'all'
			)

		# Things to prepare beforehand, since they won't be accessible
		 # when computing the rest in a single processor.
		if self.domain_type == '2D' or self.domain_type == '2D axisymmetric':
			prepare_beforehand_bmesh = [
				['entity_map', 0],
				['entity_map', 1]
			]

		elif self.domain_type == '3D':
			prepare_beforehand_bmesh = [
				['entity_map', 2]
			]

		# Unify the boundary mesh 
		unified_bmesh = UnifiedBoundaryMesh(bmesh, unified_mesh, 
			prepare_beforehand = prepare_beforehand_bmesh,
			proc_destination = 'all'
			)

		# Use the unified mesh
		fenics_mesh = unified_fenics_mesh

		# Use the unified bmesh
		bmesh = unified_bmesh

	# [FEniCS in parallel] use the unified mesh in the first processor
	if utils_fenics_mpi.runningInSerialOrFirstProcessor():
		with utils_fenics_mpi.first_processor_lock():
			# [FEniCS] Map boundary mesh to mesh
			fenics_mesh_boundary_facets = mapBoundaryMeshToMesh(bmesh, entity_type = 'facet')

			# [FEniCS] Mesh initializations
			if fenics_mesh.geometric_dimension() == 2: # 2D coordinates
				fenics_mesh.init(1, 2) # Initialize connectivity: facet (edge) <-> cell
			elif fenics_mesh.geometric_dimension() == 3: # 3D coordinates
				fenics_mesh.init(2, 3) # Initialize connectivity: facet (face) <-> cell
			else:
				raise ValueError(" ❌ ERROR: fenics_mesh.geometric_dimension() == '%s' is not defined!" %(fenics_mesh.geometric_dimension()))

			# [FEniCS] Facets and cells
			if fenics_mesh.geometric_dimension() == 2: # 2D coordinates

				fenics_mesh_coords_facets = np.array([Edge(fenics_mesh, facet_num).midpoint().array()[:-1] for facet_num in fenics_mesh_boundary_facets])
				#fenics_mesh_coords_facets = np.array([facet.midpoint().array()[:-1] for facet in facets(fenics_mesh)])

				map_index_to_cells = [Edge(fenics_mesh, facet_num).entities(2)[0] for facet_num in fenics_mesh_boundary_facets]

			elif fenics_mesh.geometric_dimension() == 3: # 3D coordinates

				fenics_mesh_coords_facets = np.array([Face(fenics_mesh, facet_num).midpoint().array()[:-1] for facet_num in fenics_mesh_boundary_facets])
				#fenics_mesh_coords_facets = np.array([facet.midpoint().array() for facet in facets(fenics_mesh)])

				map_index_to_cells = [Face(fenics_mesh, facet_num).entities(3)[0] for facet_num in FEniCS_mesh_boundary_facets]
			else:
				raise ValueError(" ❌ ERROR: fenics_mesh.geometric_dimension() == '%s' is not defined!" %(fenics_mesh.geometric_dimension()))

			map_foam_faces_to_fenics_cells = {}
			map_fenics_cells_to_foam_faces = {}
			map_foam_faces_to_fenics_cells_boundary_coords = {}

			# Adjust coordinates in case of 2D mesh
			foam_faces_center_coordinates = foam_mesh.faces_center_coordinates(boundary_name = 'all')
			foam_coords_facets = adjustCoordinatesOpenFOAMToFEniCS(foam_faces_center_coordinates, domain_type)

			# Create the maps between coordinate arrays
			#tol_mapping = 0.05*mesh_hmin # 5% of tolerance
			tol_mapping = tol_factor*mesh_hmin # Tolerance
			(map_fenics_to_foam, map_foam_to_fenics) = utils.createMapsBetweenCoordinateArrays(fenics_mesh_coords_facets, foam_coords_facets, tol_mapping = tol_mapping)

			# Go through all vertices of the OpenFOAM mesh and find the equivalent in the FEniCS mesh
			for i_foam in range(len(foam_coords_facets)):

				foam_coords = foam_coords_facets[i_foam]

				#i_fenics = utils.findNearestSubArray(fenics_mesh_coords_facets, foam_coords, parameters_to_return = 'index')
				i_fenics = map_foam_to_fenics[i_foam]

				i_fenics_entity = map_index_to_cells[i_fenics]

				map_foam_faces_to_fenics_cells[i_foam] = i_fenics_entity
				map_fenics_cells_to_foam_faces[i_fenics_entity] = i_foam

				map_foam_faces_to_fenics_cells_boundary_coords[i_foam] = fenics_mesh_coords_facets[i_fenics]

#			#### Previous version -- Surely works, but it can become quite slow in bigger meshes
#			# Go through all vertices of the OpenFOAM mesh and find the equivalent in the FEniCS mesh
#			for i_foam in range(len(foam_coords_facets)):

#				foam_coords = foam_coords_facets[i_foam]

#				i_fenics = utils.findNearestSubArray(fenics_mesh_coords_facets, foam_coords, parameters_to_return = 'index')
#				i_fenics_entity = map_index_to_cells[i_fenics]

#				map_foam_faces_to_fenics_cells[i_foam] = i_fenics_entity
#				map_fenics_cells_to_foam_faces[i_fenics_entity] = i_foam

#				map_foam_faces_to_fenics_cells_boundary_coords[i_foam] = fenics_mesh_coords_facets[i_fenics]

	else:

		map_foam_faces_to_fenics_cells = None
		map_fenics_cells_to_foam_faces = None
		map_foam_faces_to_fenics_cells_boundary_coords = None

	if utils_fenics_mpi.runningInParallel():

		# Wait for everyone!
		utils_fenics_mpi.waitProcessors() 

		# Broadcast to everyone!
		#map_foam_faces_to_fenics_cells = utils_fenics_mpi.evaluateBetweenProcessors(map_foam_faces_to_fenics_cells, operation = 'broadcast', proc_destination = 'all')
		map_foam_faces_to_fenics_cells = utils_fenics_mpi.broadcastToAll(map_foam_faces_to_fenics_cells)
		map_fenics_cells_to_foam_faces = utils_fenics_mpi.broadcastToAll(map_fenics_cells_to_foam_faces)
		map_foam_faces_to_fenics_cells_boundary_coords = utils_fenics_mpi.broadcastToAll(map_foam_faces_to_fenics_cells_boundary_coords)
	
		# Wait for everyone!
		utils_fenics_mpi.waitProcessors() 

	return map_foam_faces_to_fenics_cells, map_fenics_cells_to_foam_faces, map_foam_faces_to_fenics_cells_boundary_coords

##################### FoamVectorBoundaryToFEniCSFunction #######################

def FoamVectorBoundaryToFEniCSFunction(fenics_mesh, foam_vector, foam_mesh, map_foam_faces_to_fenics_cells, domain_type, type_of_FEniCS_Function = 'CG1', map_foam_faces_to_fenics_cells_boundary_coords = None):
	"""
	Set the values of the FoamVector boundaries to a Function in FEniCS.
	"""

	# FEniCS+MPI utilities
	from ..utils import utils_fenics_mpi

	# [FEniCS in parallel] Unified mesh
	if utils_fenics_mpi.runningInParallel():

		# Wait for everyone!
		utils_fenics_mpi.waitProcessors()

		# Local/global meshes
		local_fenics_mesh = fenics_mesh
		global_foam_mesh = foam_mesh

		# Things to prepare beforehand, since they won't be accessible
		 # when computing the rest in a single processor.
		prepare_beforehand_mesh = [
			'geometric_dimension',
			'topology',
			'cells'
		]

		# Unify the mesh
		unified_fenics_mesh = utils_fenics_mpi.UnifiedMesh(local_fenics_mesh, 
			prepare_beforehand = prepare_beforehand_mesh,
			proc_destination = 'all'
			)

		# Use the unified mesh
		fenics_mesh = unified_fenics_mesh

	# [Foam] Boundary data
	mesh_boundary_data = foam_mesh.boundaries()

	# [Foam] Boundary names
	boundary_names = list(mesh_boundary_data.keys())

	# Initialize fenics_facet_values
	if foam_vector.type == 'volVectorField':
		dim = fenics_mesh.geometric_dimension()
		fenics_boundary_cell_values = np.zeros((foam_mesh.num_cells(), dim))
	elif foam_vector.type == 'volScalarField':
		dim = 0
		fenics_boundary_cell_values = np.zeros(foam_mesh.num_cells())
	else:
		raise ValueError(" ❌ ERROR: ['%s'] type == '%s' is not (yet) available for FoamVector!" %(foam_vector.name, foam_vector.type))

	# [FEniCS in parallel] Unified mesh
	if utils_fenics_mpi.runningInParallel():

		# Back to using the local mesh
		fenics_mesh = local_fenics_mesh

	# Get the boundary values from FoamMesh
	for boundary_name in boundary_names:

		if boundary_name in ['frontSurface', 'backSurface', 'frontAndBackSurfaces']:
			pass
		else:

			startFace = mesh_boundary_data[boundary_name]['startFace']
			#startFace_boundary = startFace - foam_mesh.num_internal_faces()
			nFaces = mesh_boundary_data[boundary_name]['nFaces']

			boundary_values = foam_vector.getBoundaryValues(boundary_name)

			# No value provided
			if type(boundary_values).__name__ == 'NoneType':

				if type_of_FEniCS_Function == 'DG0':
					pass
				elif type_of_FEniCS_Function == 'CG1':
					pass

				elif type(type_of_FEniCS_Function).__name__ == 'Function': # If we have a Function, we may as well use its current values

					assert type(map_foam_faces_to_fenics_cells_boundary_coords).__name__ != 'NoneType'

					fenics_function = type_of_FEniCS_Function
					
					# Set the current values
					for i in range(nFaces):
						fenics_boundary_cell_values[map_foam_faces_to_fenics_cells[startFace + i]] = fenics_function(map_foam_faces_to_fenics_cells_boundary_coords[startFace + i])

				else:
					raise ValueError(" ❌ ERROR: type_of_FEniCS_Function == %s is not defined here!" %(type_of_FEniCS_Function))
			else:

				if foam_vector.type == 'volVectorField':
					if domain_type == '3D': 
						reshaped_boundary_values = boundary_values
					elif domain_type == '2D axisymmetric':
						reshaped_boundary_values = boundary_values
					elif domain_type == '2D': # Ignore third component
						reshaped_boundary_values = boundary_values[:,[0, 1]]
					else:
						raise ValueError(" ❌ ERROR: domain_type == %s is not defined here!" %(domain_type))
				elif foam_vector.type == 'volScalarField':
					reshaped_boundary_values = boundary_values
				else:
					raise ValueError(" ❌ ERROR: ['%s'] type == '%s' is not (yet) available for FoamVector!" %(foam_vector.name, foam_vector.type))

				# Set the new values
				for i in range(len(reshaped_boundary_values)):
					fenics_boundary_cell_values[map_foam_faces_to_fenics_cells[startFace + i]] = reshaped_boundary_values[i]

	# The function space corresponding to a 'cell' is 'DG0'
		# * DG0 = Discontinuous Galerkin (degree 0)
	DG0 = createDG0FEniCSFunctionSpace(fenics_mesh, dim = 0)

	# [FEniCS in parallel] Create the array of values for DG0
	if utils_fenics_mpi.runningInParallel():

		# Local DG0 dimension (i.e., quantity of values)
		 # * Notice that "DG0.dim()" returns the global quantity of values,
		 #   and not the local one.
		local_DG0_dim = len(Function(DG0).vector().get_local())

		# Local DG0 DoFs
		#DG0_local_DoFs = DG0.dofmap().dofs()

		# Coincidentally, DG0 'FunctionSpace' (* Not 'VectorFunctionSpace')
		 # uses the same mapping as the cells of the mesh! Also, let's 
		 # assume that no one is using ghost entities.
		#local_cell_indices = DG0_local_DoFs

		# So, let's get the cell maps from the unified mesh!
		(map_gathered_to_unified_cells, map_unified_to_gathered_cells) = unified_mesh.cell_map_gathered_unified()

		# Convert the unified cell ordering to the gathered arrays
		fenics_boundary_cell_values_unified = fenics_boundary_cell_values
		fenics_boundary_cell_values_gathered = fenics_boundary_cell_values_unified[map_unified_to_gathered_cells]

		# Get the local array from the gathered array
		fenics_boundary_cell_values_local = utils_fenics_mpi.getLocalArrayFromGatheredArray(fenics_boundary_cell_values_gathered, local_DG0_dim, dtype = fenics_mesh.cells().dtype.name)
		fenics_boundary_cell_values = fenics_boundary_cell_values_local

		# Create the local array of values for DG0
		if foam_vector.type == 'volVectorField':

			# One DG0 array for each dimension of the vector
			local_DG0_function_arrays = [np.zeros(DG0.dim()) for i in range(dim)]

			# Set the values to each DG0 array
			for i in range(dim):
				for j in range(local_DG0_dim):
					local_DG0_function_arrays[i][j] = fenics_boundary_cell_values[j][i]

			# Set to use it!
			DG0_function_arrays = local_DG0_function_arrays

		elif foam_vector.type == 'volScalarField':

			# One DG0 array
			local_DG0_function_array = np.zeros(DG0.dim())

			# Set the values
			for j in range(local_DG0_dim):
				local_DG0_function_array[j] = fenics_boundary_cell_values[j]

			# Set to use it!
			DG0_function_array = local_DG0_function_array

		else:
			raise ValueError(" ❌ ERROR: ['%s'] type == '%s' is not (yet) available for FoamVector!" %(foam_vector.name, foam_vector.type))

	else:

		# DG0 DoF map
		DG0_dofmap = DG0.dofmap()

		# Create the array of values for DG0
		if foam_vector.type == 'volVectorField':

			# One DG0 array for each dimension of the vector
			DG0_function_arrays = [np.zeros(DG0.dim()) for i in range(dim)]

			# Set the values to each DG0 array
			for i in range(dim):
				for cell in utils_fenics_mpi.get_cells(fenics_mesh):
					DG0_function_arrays[i][DG0_dofmap.cell_dofs(cell.index())] = fenics_boundary_cell_values[cell.index()][i]

		elif foam_vector.type == 'volScalarField':

			# One DG0 array
			DG0_function_array = np.zeros(DG0.dim())

			# Set the values
			for cell in utils_fenics_mpi.get_cells(fenics_mesh):
				DG0_function_array[DG0_dofmap.cell_dofs(cell.index())] = fenics_boundary_cell_values[cell.index()]

		else:
			raise ValueError(" ❌ ERROR: ['%s'] type == '%s' is not (yet) available for FoamVector!" %(foam_vector.name, foam_vector.type))

	# Create the DG0 Function
	if foam_vector.type == 'volVectorField':

		DG0_functions = [Function(DG0) for i in range(dim)]
		for i in range(dim):
			DG0_functions[i].vector().set_local(DG0_function_arrays[i])
			DG0_functions[i].vector().apply('insert')

		# Create the multidimensional DG0 function and set the values
		DG0_dim = createDG0FEniCSFunctionSpace(fenics_mesh, dim = dim)
		DG0_function = Function(DG0_dim)

		assignSubFunctionsToFunction(
			to_u_mixed = DG0_function,
			from_u_separated_array = DG0_functions
			)

	elif foam_vector.type == 'volScalarField':

		DG0_function = Function(DG0)
		DG0_function.vector().set_local(DG0_function_array)
		DG0_function.vector().apply('insert')

		DG0_dim = DG0
	else:
		raise ValueError(" ❌ ERROR: ['%s'] type == '%s' is not (yet) available for FoamVector!" %(foam_vector.name, foam_vector.type))

	# Prepare the Function to return
	if type_of_FEniCS_Function == 'DG0':
		return DG0_function

	elif type_of_FEniCS_Function == 'CG1':

		# Create the CG1 Function
		CG1 = createCG1FEniCSFunctionSpace(fenics_mesh, dim = dim)
		CG1_function = Function(CG1)

		# Create a dummy boundary condition
		#boundary_markers = MeshFunction('size_t', fenics_mesh, fenics_mesh.topology().dim() - 1)
		#boundary_markers.set_all(0)
		#
		#class Boundary_SubDomain(SubDomain):
		#	def inside(self, x, on_boundary):
		#		return on_boundary
		#
		#Boundary_SubDomain().mark(boundary_markers, 1)
		#bc_dummy = DirichletBC(CG1, DG0_function, boundary_markers, 1)

		# Create a dummy boundary condition
		bc_dummy = DirichletBC(CG1, DG0_function, 'on_boundary')

		# Set the values on the boundaries
		bc_dummy.apply(CG1_function.vector())

		return CG1_function

	elif type(type_of_FEniCS_Function).__name__ == 'Function':

		fenics_function = type_of_FEniCS_Function
		function_space = fenics_function.function_space()

		# Create a dummy boundary condition
		bc_dummy = DirichletBC(function_space, DG0_function, 'on_boundary')

		# Set the values on the boundaries
		bc_dummy.apply(fenics_function.vector())

		return fenics_function

	else:
		raise ValueError(" ❌ ERROR: type_of_FEniCS_Function == %s is not defined here!" %(type_of_FEniCS_Function))

################################################################################
######################## Manipulation of FEniCS DoF's ##########################
################################################################################

###################### assignSubFunctionsToFunction ############################

def assignSubFunctionsToFunction(
	to_u_mixed = None,
	from_u_separated_array = []
	):
	"""
	Assign from "u_separated_array" to "u_mixed".
	"""

	u_mixed = to_u_mixed
	u_separated_array = from_u_separated_array

	U_mixed = u_mixed.function_space()

	do_it_separate = False
	for i in range(len(u_separated_array)):

		if type(u_separated_array[i]).__name__ == 'NoneType': # If there is None in "u_separated_array"
			do_it_separate = True
			break
		elif type(u_separated_array[i]).__name__ == 'Indexed':
			do_it_separate = True
			break
		else:
			pass

	if do_it_separate == True:

		(U_collapsed, u_collapsed) = getSubfunctions(u_mixed, zero_the_functions = False)

		U_separated_array = []
		u_separated_array = []
		for i in range(len(from_u_separated_array)):
			if type(from_u_separated_array[i]).__name__ == 'NoneType': # If there is None in "u_separated_array"

				u_collapsed_index = i
				u_separated_array += [u_collapsed[u_collapsed_index]]
				U_separated_array += [U_collapsed[u_collapsed_index]]

			elif type(from_u_separated_array[i]).__name__ == 'Indexed':
				current_u = from_u_separated_array[i]
				full_u = current_u.ufl_operands[0]
				full_u_index = current_u.ufl_operands[1].indices()[0]._value

				(U_sep, u_sep) = getSubfunctions(full_u, zero_the_functions = False)

				u_separated_array += [u_sep[full_u_index]]
				U_separated_array += [U_sep[full_u_index]]
			else:
				u_separated_array += [from_u_separated_array[i]]
				U_separated_array += [u_separated_array[i].function_space()]
	else:
		U_separated_array = [u_separated_array[i].function_space() for i in range(len(u_separated_array))]

	if U_mixed.num_sub_spaces() == 0 and len(U_separated_array) == 1 and U_separated_array[0].num_sub_spaces() == 0:
		U_separated_array = U_separated_array[0]
		u_separated_array = u_separated_array[0]

	# Create FunctionAssigner
	assigner = FunctionAssigner(U_mixed, U_separated_array)

	# Assign from "u_separated_array" to "u_mixed"
	assigner.assign(u_mixed, u_separated_array)

################################# getSubfunctions ##############################

def getSubfunctions(u, zero_the_functions = False):
	"""
	Returns the zeroed subfunctions of u.
	"""

	U = u.function_space()
	num_subspaces = U.num_sub_spaces()
	U_collapsed = [U.sub(i).collapse() for i in range(num_subspaces)]

	if zero_the_functions == True:
		u_collapsed = [Function(U_collapsed[i]) for i in range(len(U_collapsed))]
	else:
		u_collapsed_ = u.split(deepcopy = True)
		u_collapsed = [Function(U_collapsed[i]) for i in range(len(U_collapsed))]

		for i in range(len(u_collapsed)):
			setFunctionFromFunction(u_collapsed[i], u_collapsed_[i])

	return U_collapsed, u_collapsed

########################### setFunctionFromFunction ############################

def setFunctionFromFunction(u_destination, u_origin):
	"""
	Set a Function to another Function.
	"""
	u_origin_array = u_origin.vector().get_local()
	u_destination.vector().set_local(u_origin_array)
	u_destination.vector().apply('insert')

########################## convertUFLToDG0Function #############################

def convertUFLToDG0Function(mesh, ufl_something, projection_setup):
	"""
	Convert something in UFL to a DG0 Function.
	"""

	try:
		ufl_shape = ufl_something.ufl_shape
		if len(ufl_shape) == 0:
			DG0 = FunctionSpace(mesh, 'DG', 0)
		elif len(ufl_shape) == 1:
			dim_ = ufl_shape[0]
			DG0 = VectorFunctionSpace(mesh, 'DG', 0, dim = dim_)
		else:
			raise ValueError(" ❌ ERROR: ufl_shape == %s from '%s' is not recognized!" %(ufl_shape, ufl_something))

		projected_var = configuredProject(ufl_something, DG0, projection_setup)
		return projected_var

	except:
		import traceback
		traceback.print_exc()
		utils.customPrint(" ❗ Unable to automatically convert to a Function!")
		return None

################################ filterVariable ################################

@no_annotations()
def filterVariable(var_to_filter, function_space, domain_type = '2D', filter_parameters = {}, overload_variable = False, type_of_variable = 'scalar'):
	"""
	Filters a variable by using a Helmholtz filter.
	-> Use this if you think that the projected variable became overly "chopped",
	   ruining other post-processings you wanted to perform in FEniCS.
	"""

	# Input parameters
	filter_radius_divider = filter_parameters.get('filter_radius_divider', 10.)
	quadrature_degree = filter_parameters.get('quadrature_degree', None) # None == Automatic
	solver_parameters = filter_parameters.get('solver_parameters', {'linear_solver': 'default', 'preconditioner': 'default'})

	# Start!
	utils.customPrint(" 🌊 Applying Helmholtz filter for a smoother variable (filter_radius_divider = %e)..." %(filter_radius_divider))

	# Mesh
	mesh = function_space.mesh()

	# Domain multiplier and differential operator
	if domain_type == '2D axisymmetric':
		(r_, z_) = SpatialCoordinate(mesh); nr = 0; nz = 1

		domain_multiplier = r_

		if type_of_variable == 'scalar':
			grad_ = lambda a: as_tensor([Dx(a, nr), 0, Dx(a, nz)])

		elif type_of_variable == 'vector':
			def grad_(var):
				[ar, atheta, az] = split(var)
				return as_tensor([[Dx(ar, nr),		-atheta/r_,	Dx(ar, nz)],
						   [Dx(atheta, nr),	ar/r_,		Dx(atheta, nz)],
						   [Dx(az, nr),		0,		Dx(az, nz)]])
		else:
			raise ValueError(" ❌ ERROR: type_of_variable == '%s' is not defined!" %(type_of_variable))

	else:
		domain_multiplier = 1.0
		grad_ = lambda a: grad(a)

	# Filter radius
	from ..utils import utils_fenics_mpi
	r_H = function_space.mesh().hmin()/filter_radius_divider
	if utils_fenics_mpi.runningInParallel():
		r_H = utils_fenics_mpi.evaluateBetweenProcessors(r_H, operation = 'minimum', proc_destination = 'all')
	#r_H = function_space.mesh().hmax()/filter_radius_divider
	#if utils_fenics_mpi.runningInParallel():
	#	r_H = utils_fenics_mpi.evaluateBetweenProcessors(r_H, operation = 'maximum', proc_destination = 'all')

	# Test function
	w_H = TestFunction(function_space)

	# The solution
	var_filtered_trial = TrialFunction(function_space)

	# Weak form of the modified Helmholtz equation
	F = (

		# + ∫_Π r_H² ∇var_new•∇w_H dΠ
		r_H**2 * inner(grad_(var_filtered_trial), grad_(w_H))

		# + ∫_Π var_new•w_H dΠ
		+ inner( var_filtered_trial , w_H )

		# - ∫_Π var•w_H dΠ
		- inner( var_to_filter , w_H )

		)*domain_multiplier*dx

	# Prepare variable to store solution
	if overload_variable == True:
		var_filtered = var_to_filter
	else:
		var_filtered = Function(function_space)

	# Solve the linear system
	a = lhs(F)
	L = rhs(F)
	solve(a == L, var_filtered, solver_parameters = solver_parameters, form_compiler_parameters = {"quadrature_degree": quadrature_degree})

	# Return the filtered variable
	return var_filtered


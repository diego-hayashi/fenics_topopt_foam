################################################################################
#                                 FoamVector                                   #
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

# FEniCS+MPI utilities
from ..utils import utils_fenics_mpi

################################## FoamVector ##################################

class FoamVector():
	"""
	Vector variable for OpenFOAM.
	It is stored in [problem_folder]/[time_step_name]/[name].
	It can be a 'volScalarField' or a 'volVectorField'.

	-> Sorry that I did not envision this variable as a high-level variable
	   as a FEniCS Function. This class would be more like an (extremely 
	   simplified) PETSc Vector with additional specific functionality.

	"""

	def __init__(self, foam_mesh, data):

		# Data (from FoamReader)
		self.reloadData(data, foam_mesh)

	############################ Apply changes #############################

	def set_no_changes_to_apply(self):
		"""
		Set that it is not to apply changes.
		"""
		self._APPLY_CHANGES = False

	def check_if_to_apply_changes(self):
		"""
		Return if it is to apply changes
		"""
		return self._APPLY_CHANGES

	def set_to_apply_changes(self, name):
		"""
		Set to apply all changes to the file.
		* You CANNOT forget to use this in the end. Otherwise, the changes
		  will NOT be applied to the FoamFile that is used by OpenFOAM!
		"""

		if name == 'insert':
			self._APPLY_CHANGES = True
		else:
			raise ValueError(" ❌ ERROR: name == '%s' is not defined!" %(name))

	############################### reloadData #############################

	def reloadData(self, data, foam_mesh):
		"""
		Set the data obtained from the FoamReader.
		"""

		# FoamMesh
		self.foam_mesh = foam_mesh

		# Size
		self.size = self.foam_mesh.num_cells()

		# Data
		self.data = data

		# Check class
		self.type = self.data['FoamFile']['class']
		if self.type not in ['volVectorField', 'volScalarField']:
			raise ValueError(" ❌ ERROR: ['%s'] type == '%s' is not (yet) available for FoamVector!" %(self.name, self.type))

		# Array of values
		array = self.data['internalField']

		if self.type == 'volVectorField':
			self.num_components = 3
			if len(array.shape) == 1:
				# A uniform value has been imposed
				array_to_set = np.array([array]*self.size)
			else:
				# A nonuniform value has been imposed
				array_to_set = array

		elif self.type == 'volScalarField':
			self.num_components = 1
			if len(array) == 1:
				# A uniform value has been imposed
				array_to_set = np.array([array[0]]*self.size)
			else:
				# A nonuniform value has been imposed
				array_to_set = array

		self.data['internalField'] = array_to_set

		# Additional initialization
		self._local_array = None

		# Flag that indicates if it is to apply changes
		self._APPLY_CHANGES = False

	######################### Boundary conditions ##########################

	def setBoundaryCondition(self, boundary_name, bc_type, value, other_parameters = {}):
		"""
		Set a boundary condition in an existent boundary_name.
		"""

		if boundary_name not in self.data['boundaryField']:
			raise ValueError(" ❌ ERROR: [%s] Boundary '%s' is not available in boundaryField == %s!" %(self.name, boundary_name, self.data['boundaryField']))

		self.addBoundaryCondition(boundary_name, bc_type, value, other_parameters = other_parameters)

	def addBoundaryCondition(self, boundary_name, bc_type, value, other_parameters = {}, include_if_not_already_included = False):
		"""
		Add boundary condition.
		"""

		if include_if_not_already_included == True:
			utils.customPrint(" ❗ Boundary condition '%s' already exists. Skipping..." %(boundary_name))
		else:

			# Clear boundary condition
			self.data['boundaryField'][boundary_name] = {}

			# Set the type
			self.data['boundaryField'][boundary_name]['type'] = bc_type

			# Set the value
			if type(value).__name__ == 'NoneType':
				if 'value' in self.data['boundaryField'][boundary_name]:
					del self.data['boundaryField'][boundary_name]['value']
			else:

				if (
					(len(value) == self.num_components)
					or (len(value) == self.foam_mesh.boundary(boundary_name)['nFaces'])
					):
					pass
				else:
					raise ValueError(" ❌ ERROR: [%s] Boundary '%s' is of length %d (uniform size: %d), while trying to impose length %d!" %(self.name, boundary_name, self.foam_mesh.boundary(boundary_name)['nFaces'], self.num_components, len(value)))

				self.data['boundaryField'][boundary_name]['value'] = value

			# Set the other parameters

			assert 'value' not in other_parameters
			assert 'type' not in other_parameters

			for key in other_parameters:
				self.data['boundaryField'][boundary_name][key] = other_parameters[key]

	def getBoundaryValues(self, boundary_name):
		"""
		Get the boundary values from an existent boundary_name.
		"""

		if boundary_name not in self.data['boundaryField']:
			raise ValueError(" ❌ ERROR: [%s] Boundary '%s' is not available in boundaryField == %s!" %(self.name, boundary_name, self.data['boundaryField']))

		array = self.data['boundaryField'][boundary_name].get('value', None)

		if type(array).__name__ != 'NoneType':

			boundary_data = self.foam_mesh.boundary(boundary_name)

			nFaces = boundary_data['nFaces']

			if self.num_components == 3:

				if len(array.shape) == 1:
					# A uniform value has been imposed
					values = np.array([array]*nFaces)
				else:
					# A nonuniform value has been imposed
					values = array

			elif self.num_components == 1:

				if len(array) == 1:
					# A uniform value has been imposed
					values = np.array([array[0]]*nFaces)
				else:
					# A nonuniform value has been imposed
					values = array
		else:
			values = array

		return values

	def setBoundaryConditionValue(self, boundary_name, value):
		"""
		Set a value for the boundary condition.
		"""

		if boundary_name not in self.data['boundaryField']:
			raise ValueError(" ❌ ERROR: [%s] Boundary '%s' is not available in boundaryField == %s!" %(self.name, boundary_name, self.data['boundaryField']))

		self.data['boundaryField'][boundary_name]['value'] = value

	#################### getMaxMinValuesFromBoundaryType ###################

	def getMaxMinValuesFromBoundaryType(self, boundary_type):
		"""
		Returns the maximum and minimum value on a boundary type (for example, 'wall').
		-> This may be useful if the FoamVector is yPlus (i.e., y⁺).
		"""

		# Get the boundaries 
		boundaries = self.foam_mesh.boundaries()

		# Get the maximum value
		max_value = None
		min_value = None
		for boundary_name in boundaries:
			boundary = boundaries[boundary_name]
			if boundary['type'] == boundary_type:
				value = self.data['boundaryField'][boundary_name]['value']

				if type(value).__name__ == 'ndarray':

					if self.type == 'volVectorField':
						new_max_value = value.max(axis = 0)
						new_min_value = value.min(axis = 0)
					elif self.type == 'volScalarField':
						new_max_value = value.max()
						new_min_value = value.min()

					if type(max_value).__name__ == 'NoneType':
						max_value = new_max_value
						min_value = new_min_value
					else:
						max_value = np.maximum(max_value, new_max_value)
						min_value = np.maximum(min_value, new_min_value)

		return max_value, min_value

	########################## Array manipulation ##########################

	def get_local(self):
		"""
		Returns a copy of the local array of values.
		"""
		global_array = self.get_global()

		if utils_fenics_mpi.runningInParallel():

			(global_to_local_DoF_map, local_to_global_DoF_map) = self.foam_mesh.get_local_mapping()

			assert type(global_to_local_DoF_map).__name__ != 'NoneType', " ❌ ERROR: Local mapping has still not been defined through FoamMesh.set_local_mapping"

			local_array = global_array[global_to_local_DoF_map]
		else:
			local_array = global_array

		self._local_array = local_array

		return local_array

	def set_local(self, new_array, apply_global_changes = True):
		"""
		Sets the local array of values.
		-> Don't forget to run "foam_vector.set_to_apply_changes('apply')" later.
		"""

		local_array = new_array

		if utils_fenics_mpi.runningInParallel():

			assert apply_global_changes == True, " ❌ ERROR: All changes are applied globally, sorry."

			# Wait for everyone!
			utils_fenics_mpi.waitProcessors()

			# Local DoF map
			(global_to_local_DoF_map, local_to_global_DoF_map) = self.foam_mesh.get_local_mapping()

			assert type(global_to_local_DoF_map).__name__ != 'NoneType', " ❌ ERROR: Local mapping has still not been defined through FoamMesh.set_local_mapping"

			# Zero global array
			global_array = np.zeros_like(self.data['internalField'])

			# Set the local values
			global_array[global_to_local_DoF_map] = local_array

			# Wait for everyone!
			utils_fenics_mpi.waitProcessors()

			# Apply the changes globally
			global_array = utils_fenics_mpi.evaluateBetweenProcessors(global_array, operation = 'term-by-term sum of arrays', proc_destination = 'all', axis = 0)

			# Wait for everyone!
			utils_fenics_mpi.waitProcessors()

			# Set the global values
			self.set_global(global_array)

			# Wait for everyone!
			utils_fenics_mpi.waitProcessors()

		else:

			# Global array
			global_array = local_array

			# Set the global values
			self.set_global(global_array)

		self._local_array = None

	def get_global(self):
		"""
		Returns a copy of the global array of values.
		"""
		return self.data['internalField'].copy()

	def set_global(self, new_array):
		"""
		Sets the global array of values.
		-> Don't forget to run "foam_vector.set_to_apply_changes('apply')" later.
		"""

		assert len(new_array) == self.size, " ❌ ERROR: (len(new_array) = %s) != (self.size = %s), new_array = %s" %(len(new_array), self.size, new_array)
		self.data['internalField'] = new_array
		self._local_array = None

	############################### Values #################################

	def __call__(self, x):
		"""
		Return the value of the variable in coordinate x.
		"""
		raise NotImplementedError(" ❌ ERROR: Not implemented yet. For this, we would need the coordinates of all the nodes for all the cells, and a nice algorithm to search for the location of the point. Please use self.value_from_index or self.nearest_nodal_value and convert the variable to a DG0 variable in FEniCS, because FEniCS has already got this implemented!")

	def value_from_index(self, var_index, use_local_indexing = True):
		"""
		Return the value of the variable in index var_index.
		* This is useful when we just want to pass the values to another
		  type of data structure in which the ordering of nodes is the same.
		"""
		if utils_fenics_mpi.runningInParallel() and (use_local_indexing == True):
			if type(self._local_array).__name__ == 'NoneType':
				local_array = self.get_local()
			else:
				local_array = self._local_array
			return local_array[var_index]
		else:
			return self.data['internalField'][var_index]

	def nearest_nodal_value(self, x, tol = 1E-10):
		"""
		Return the nearest value of the variable in coordinate x.
		* This is useful when we just want to pass the values to another
		  type of data structure in which small numerical errors may occur.
		"""

		self.coordinates = foam_mesh.cell_centers()

		# Find the index of the nearest coordinate
		var_index = utils.findNearestSubArray(x, self.coordinates, parameters_to_return = 'index')

		distance = np.linalg.norm(self.coordinates[var_index] - x, axis = 1)

		if distance > tol:
			raise ValueError(""" ❌ ERROR: Distance of found coordinate (%e) outside tolerance (%e)!
Searching for coordinate: x = %s
Coordinate found in mesh: [%d] %s
""" % 			(
			distance, tol,
			x,
			var_index, self.coordinates[var_index],
			))

		return self.value_from_index[var_index]

	################################# name #################################

	@property
	def name(self):
		"""
		Name of the FoamVector.
		"""
		return self.data['FoamFile']['object']


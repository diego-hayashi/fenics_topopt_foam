################################################################################
#                                   FoamMesh                                   #
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

################################### FoamMesh ###################################

class FoamMesh():
	"""
	The mesh files are stored in: [problem_folder]/constant/polyMesh

	* Another option of implementation would be by using Ofpp (OpenFOAM Python Parser)
	  ( https://github.com/dayigu/ofpp ). However, the functionality 
	  (i.e., data that can be provided) is quite restricted and the data 
	  provided is not in a format that is easy for manipulating in this 
	  code here (* In my opinion, using dictionaries (instead of tuples) makes 
	  it easier to provide more mesh data as needed).

	* Check some more functionality that can be added here in https://cfd.direct/openfoam/user-guide/v7-post-processing-cli/#x32-2390006.2

	* Small trivia:
		Why are the names "repeated" in part of the data, as in "self.data['points']['points']"?
		-> Because, in FoamReader:
			First 'points'  = File name
			Second 'points' = Variable name
		-> This only happens when there is only a single array inside the file, because
			OpenFOAM does not give a name for the array only in this situation.

	"""

	def __init__(self, data, domain_type = 'unset'):

		self.reloadData(data)
		self._domain_type = domain_type

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

	def reloadData(self, data):
		"""
		Set the data obtained from the FoamReader.
		"""

		self.data = data
		self.problem_folder = self.data['problem_folder']
		self.foam_reader = self.data['FoamReader']

		# Leave some calculations for when needed
		self._cell_centers = None
		self._cell_volumes = None
		self._faces_coordinates = None
		self._faces_center_coordinates = None

		# Leave it to be set when needed
		self._global_to_local_DoF_map = None
		self._local_to_global_DoF_map = None

		# Flag that indicates if it is to apply changes
		self._APPLY_CHANGES = False

	############################# DoF mapping ##############################

	def set_local_mapping(self, global_to_local_DoF_map, local_to_global_DoF_map):
		"""
		Set the local mapping.
		"""
		self._global_to_local_DoF_map = global_to_local_DoF_map
		self._local_to_global_DoF_map = local_to_global_DoF_map

	def get_local_mapping(self):
		"""
		Get the local mapping.
		"""
		return self._global_to_local_DoF_map, self._local_to_global_DoF_map

	############################# Coordinates ##############################

	def coordinates(self):
		"""
		Coordinates of the mesh.
		"""
		return self.data['points']['points']

	############################# Boundaries ###############################

	def boundaries(self):
		"""
		Get the boundaries of the mesh.
		"""
		return self.data['boundary']['boundary']

	def boundary(self, boundary_name):
		"""
		Get a boundary of the mesh.
		"""
		return self.data['boundary']['boundary'][boundary_name]

	############################### Counts #################################

	def num_cells(self):
		"""
		Number of cells.
		"""
		return max(self.data['owner']['owner']) + 1

	def num_vertices(self):
		"""
		Number of vertices.
		"""
		return len(self.data['points']['points'])

	def num_faces(self):
		"""
		Number of faces.
		"""
		return len(self.data['faces']['faces'])

	def num_internal_faces(self):
		"""
		Number of internal faces.
		"""
		return len(self.data['neighbour']['neighbour'])

	############################### Faces ##################################

	def faces_indices(self):
		"""
		Indices of the faces of the mesh.
		"""
		return self.data['faces']['faces']

	def faces_coordinates(self):
		"""
		Coordinates of the vertices of the faces of the mesh.
		"""

		if type(self._faces_coordinates).__name__ == 'NoneType':

			faces_indices = self.faces_indices()
			coordinates = self.coordinates()

			self._faces_coordinates = utils.create_array_from_list_of_arrays([np.array([coordinates[vertex_index] for vertex_index in face_indices], dtype = 'float') for face_indices in faces_indices])

		return self._faces_coordinates.copy()

	def faces_center_coordinates(self, boundary_name = 'all'):
		"""
		Coordinates of the centers of the faces of the mesh.
		"""

		if type(self._faces_coordinates).__name__ == 'NoneType':

			faces_coordinates = self.faces_coordinates()

			self._faces_center_coordinates = np.array([utils.computeCenterCoordinates(face_coordinates) for face_coordinates in faces_coordinates])

		if boundary_name == 'all':
			faces_center_coordinates = self._faces_center_coordinates.copy()
		else:
			boundary = self.data['boundary']['boundary'][boundary_name]
			nFaces = boundary['nFaces']
			startFace = boundary['startFace']

			faces_indices = np.arange(startFace, startFace + nFaces)
			faces_center_coordinates = np.take(self._faces_center_coordinates, faces_indices, axis = 0)
				# https://stackoverflow.com/questions/14162026/how-to-get-the-values-from-a-numpy-array-using-multiple-indices
				# https://docs.scipy.org/doc/numpy/reference/generated/numpy.take.html

		return faces_center_coordinates

	############################### Cells ##################################

	def cell_centers(self, compute_2D_cell_centers_for_triangles_if_2D_axisymmetric = False, reuse_if_possible = True):
		"""
		Compute the cell centers.
		"""

		if (type(self._cell_centers).__name__ == 'NoneType') or (reuse_if_possible == False):

			if (compute_2D_cell_centers_for_triangles_if_2D_axisymmetric == True) and (self.domain_type == '2D axisymmetric'):

				# This adjustment is important for avoiding issues when mapping
				 # from OpenFOAM to FEniCS: Since the FEniCS mesh is 2D, the centers
				 # of the cells in OpenFOAM may feature some sort of displacement with respect
				 # to the 2D centers in FEniCS, which may be large for coordinates that are too close to the symmetry axis.

				# Save the 2D cell center coordinates (in the first time step) to [problem_folder]/0/cellCenters2Dtriangles
				#time = 0 # Write to the "0" folder
				if utils_fenics_mpi.runningInSerialOrFirstProcessor():
					with utils_fenics_mpi.first_processor_lock():
						utils.customPrint("\n 🌀 Computing 2D cell centers of the 2D axisymmetric mesh (ATTENTION: ONLY works for triangular meshes)...")
						utils.run_command_in_shell('cellCenters2Dtriangles -case %s -constant -time constant' %(self.problem_folder), mode = 'print directly to terminal', indent_print = True)

				# Read the cell center data
				utils.customPrint("\n 🌀 Reading 2D cell centers of the 2D axisymmetric mesh (ATTENTION: ONLY works for triangular meshes)...")
				CellCenter_data = self.foam_reader.readMeshData('../cellCenters2Dtriangles')

				# Load the array of cell centers
				self._cell_centers = CellCenter_data['internalField']

			else:
				# Save the cell center coordinates (in the first time step) to [problem_folder]/0/C
				#time = 0 # Write to the "0" folder
				if utils_fenics_mpi.runningInSerialOrFirstProcessor():
					with utils_fenics_mpi.first_processor_lock():
						utils.customPrint("\n 🌀 Computing cell centers of the mesh...")
						utils.run_command_in_shell('postProcess -case %s -func \'writeCellCentres\' -constant -time constant' %(self.problem_folder), mode = 'print directly to terminal', indent_print = True)

				# Read the cell center data
				utils.customPrint("\n 🌀 Reading cell centers of the mesh...")
				CellCenter_data = self.foam_reader.readMeshData('../C')

				# Load the array of cell centers
				self._cell_centers = CellCenter_data['internalField']

		return self._cell_centers.copy()

	def cell_volumes(self):
		"""
		Compute the cell volumes.
		"""

		if type(self._cell_volumes).__name__ == 'NoneType':

			# Save the cell center coordinates (in the first time step) to [problem_folder]/0/Cx
			#time = 0 # Write to the "0" folder
			if utils_fenics_mpi.runningInSerialOrFirstProcessor():
				with utils_fenics_mpi.first_processor_lock():
					utils.customPrint("\n 🌀 Computing cell volumes of the mesh...")
					utils.run_command_in_shell('postProcess -case %s -func \'writeCellVolumes\' -constant -time constant' %(self.problem_folder), mode = 'print directly to terminal', indent_print = True)

			# Read the cell volume data
			utils.customPrint("\n 🌀 Reading cell volumes of the mesh...")
			CellVolume_data = self.foam_reader.readMeshData('../V')

			# Load the array of cell volumes
			self._cell_volumes = CellVolume_data['internalField']

		return self._cell_volumes.copy()

	################################# name #################################

	@property
	def name(self):
		"""
		Name of the FoamMesh.
		"""
		return "polyMesh"

	############################## domain_type #############################

	@property
	def domain_type(self):
		"""
		Name of the domain type.
		"""
		if self._domain_type == 'unset':
			raise ValueError(" ❌ ERROR: domain_type is unset! Please, set it before using!")

		assert self._domain_type in ['3D', '2D', '2D axisymmetric'], " ❌ ERROR: self._domain_type == '%s' is undefined!" %(self._domain_type)

		return self._domain_type


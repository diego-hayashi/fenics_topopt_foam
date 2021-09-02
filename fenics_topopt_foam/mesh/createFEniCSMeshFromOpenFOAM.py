################################################################################
#                         createFEniCSMeshFromOpenFOAM                         #
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

# Mesh
from ..mesh.exportMesh import exportMesh

# FoamWriter
from ..io.FoamMeshWriter import FoamMeshWriter

# I/O (Input/Output)
from ..io.FoamReader import FoamReader

# Utilities
from ..utils import utils

# FEniCS utilities
from ..utils import utils_fenics

# FEniCS+MPI utilities
from ..utils import utils_fenics_mpi

############################### FEniCS libraries ###############################

try:
	from fenics import *
	import fenics
	 # Do not import dolfin-adjoint here. This is a non-annotated module!
except:
	utils.printDebug(" ❌ FEniCS not installed! Can't use FEniCSFoamSolver!")

####################### createFEniCSMeshFromOpenFOAM ###########################

def createFEniCSMeshFromOpenFOAM(problem_folder, domain_type = '2D', time_step_name = '0', 
	return_boundary_data = True,
	use_dolfin_adjoint_if_exists = True
	):
	"""
	Creates a mesh in FEniCS from a mesh in OpenFOAM.

	-> Remember:
		2D meshes require:
			1) Two planes, in the interval of z coordinates given by {-t, +t}
			2) The two faces MUST be marked as 'frontAndBackSurfaces'
		2D axisymmetric meshes require:
			1) Two planes in the form of a wedge, in the interval of z coordinates given by {-t.r/r_max, +t.r/r_max}
			2) The two faces MUST be marked as 'frontSurface' (for +t.r/r_max) and as 'backSurface' (for -t.r/r_max)

	-> Opening an adjusted .xdmf file in FEniCS for obtaining the mesh.
	   Only supports triangular elements as of now, because I do not
	   know an easy way to import a generic .xdmf file into a FEniCS 
	   quadrilateral mesh (at least for now, in the current stable FEniCS version).

	"""

	# [Parallel] Wait for everyone!
	if utils_fenics_mpi.runningInParallel():
		utils_fenics_mpi.waitProcessors()

	# Reset matching elements search
	utils.resetSearchMatchingElements()

	################### Export the mesh to VTK format ######################

	# Export mesh to VTK
	exportMesh(problem_folder, mesh_file_type = 'VTK', time_step_name = time_step_name)

	# VTK folder created by OpenFOAM
	VTK_folder = "%s/VTK" %(problem_folder)

	# Get the .vtk file name
	vtk_file_names = utils.getNamesOfFilesInFolder(VTK_folder, file_extensions_to_consider = ['%s.vtk' %(time_step_name)])

	if len(vtk_file_names) == 1:
		vtk_file_name = vtk_file_names[0]
	else:
		raise ValueError(" ❌ ERROR: There are %d %s.vtk files in '%s'." %(len(vtk_file_names), time_step_name, VTK_folder))

	# Wait for everyone!
	if utils_fenics_mpi.runningInParallel():
		utils_fenics_mpi.waitProcessors()

	if utils_fenics_mpi.runningInSerialOrFirstProcessor():

		with utils_fenics_mpi.first_processor_lock():

			########### Read and edit the mesh with meshio ##########

			# meshio
			import meshio

			# Read mesh into the meshio format
			utils.customPrint(" 🌀 [meshio] Reading '%s' to meshio..." %("%s/%s" %(VTK_folder, vtk_file_name)))

			if int(float(meshio.__version__.split('.')[0])) >= 4:
				meshio_mesh = meshio.read("%s/%s" %(VTK_folder, vtk_file_name), file_format = 'vtk')
			else:
				meshio_mesh = meshio.read("%s/%s" %(VTK_folder, vtk_file_name), file_format = 'vtk-ascii')

			# Clear additional data from the .vtk file, because we only need the mesh
			 # * Unfortunately, boundary information still can not be written to the mesh file (lacking implementation).
				# https://github.com/nschloe/meshio/issues/175 

			utils.customPrint(" 🌀 [meshio] Clearing up data that do not belong to the mesh...")
			meshio_mesh.cell_data = {}
			meshio_mesh.element_sets = {}
			meshio_mesh.field_data = {}
			meshio_mesh.node_sets = {}
			meshio_mesh.point_data = {}

			utils.customPrint(" 🌀 [meshio] Preparing vertices and cells...")
			if domain_type == '2D' or domain_type == '2D axisymmetric':

				# Points and cells
				mesh_vertices = meshio_mesh.points
				if int(float(meshio.__version__.split('.')[0])) >= 4:
					#mesh_cells_dict = meshio_mesh.cells_dict
					mesh_cells_dict = {meshio_cell.type : meshio_cell.data for meshio_cell in meshio_mesh.cells}

				else:
					mesh_cells_dict = meshio_mesh.cells

				if domain_type == '2D': # "z" column (column number 2)

					# Front surface indices
					frontSurface_vertex_indices = np.where(mesh_vertices[:,2] >= 0)[0]
					assert len(frontSurface_vertex_indices) > 0 # Sorry, must have for 2D mesh

					# Back surface indices
					backSurface_vertex_indices = np.where(mesh_vertices[:,2] < 0)[0]
					assert len(backSurface_vertex_indices) > 0 # Sorry, must have for 2D mesh

				elif domain_type == '2D axisymmetric': # "θ" column (column number 1)

					# Front surface indices
					frontSurface_vertex_indices = np.where(mesh_vertices[:,1] >= 0)[0]
					assert len(frontSurface_vertex_indices) > 0 # Sorry, must have for 2D axisymmetric mesh

					# Back surface indices
					backSurface_vertex_indices = np.where(mesh_vertices[:,1] < 0)[0]
					assert len(backSurface_vertex_indices) > 0 # Sorry, must have for 2D axisymmetric mesh

				# Map frontSurface vertex indices
				map_from_frontSurface_vertex_indices_to_new_vertices = utils_fenics.invertMapOrder(frontSurface_vertex_indices.tolist())

				# Number of new vertices
				num_new_vertices = len(mesh_vertices) - len(backSurface_vertex_indices)
				if domain_type == '2D':
					# x, y, z

					# Create a copy and then trim the "z" column (column number 2)
					mesh_vertices_copy = mesh_vertices.copy()[:,[0,1]]

				elif domain_type == '2D axisymmetric':
					# r, θ, z

					# Create a copy and then trim the "θ" column (column number 1)
					mesh_vertices_copy = mesh_vertices.copy()[:,[0,2]]

				# New vertices
				new_vertices = mesh_vertices_copy[frontSurface_vertex_indices]

				def createNewEntityFromVertices(entity_vertices):
					"""
					Creates a new entity from vertices.
					"""

					# Keep only the frontSurface vertices
					frontSurface_entity_vertices = utils.searchMatchingElements(entity_vertices, frontSurface_vertex_indices, reuse_previous_dictionary = True, profile_key = 'create new cell')

					# Map all vertices
					new_entity_vertices = np.array([map_from_frontSurface_vertex_indices_to_new_vertices[vertex] for vertex in frontSurface_entity_vertices], dtype = 'int32')

					if len(new_entity_vertices) == 4: # quad
						# The order of the two last vertices must be inverted if we are using quad.
						 # This way, the vertices follow a loop around the cell. This is the case of, for example,
						 #
						 # new_entity_vertices = [1, 2, 4, 3]
						 # 
						 # 4                  2
						 #  ● ────────────── ●
						 #  │                │
						 #  │                │
						 #  │                │
						 #  │                │
						 #  │                │
						 #  │                │
						 #  │                │
						 #  ● ────────────── ●
		 				 # 3                  1
						 # 
		 				 # => We must invert the vertices 4 and 3, obtaining new_entity_vertices = [1, 2, 3, 4] in the end!
						 # 
						new_entity_vertices = np.array([new_entity_vertices[0], new_entity_vertices[1], new_entity_vertices[3], new_entity_vertices[2]], dtype = 'int32')

					return new_entity_vertices

				new_mesh_cells_dict = {}
				for mesh_cell_type in mesh_cells_dict:
					mesh_cells = mesh_cells_dict[mesh_cell_type]

					# Set the cell type
						# >> meshio/_common.py
						# https://github.com/nschloe/meshio/issues/630
						# https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
					if mesh_cell_type == 'hexahedron': # 'hexahedron' in OpenFOAM
						new_cell_type = 'quad'

					elif mesh_cell_type == 'tetrahedron': # 'tetrahedron' in OpenFOAM
						new_cell_type = 'triangle'

					elif mesh_cell_type == 'wedge': # 'prism' in OpenFOAM
						if 'hexahedron' in mesh_cells_dict:
							new_cell_type = 'quad'
						else:
							new_cell_type = 'triangle'

					elif mesh_cell_type == 'pyramid': # 'pyramid' in OpenFOAM
						if 'hexahedron' in mesh_cells_dict:
							new_cell_type = 'quad'
						else:
							new_cell_type = 'triangle'

					else:
						raise ValueError(" ❌ ERROR: Cell type '%s' is currently not being considered in this function." %(mesh_cell_type))

					# Create new entry
					if new_cell_type not in new_mesh_cells_dict:
						new_mesh_cells_dict[new_cell_type] = []
					else:
						pass

					# Include all cells
					new_mesh_cells_dict[new_cell_type] += [createNewEntityFromVertices(cell) for cell in mesh_cells]

				# To finish up, let's convert to NumPy array
				for mesh_cell_type in new_mesh_cells_dict:
					new_mesh_cells_dict[mesh_cell_type] = np.array(new_mesh_cells_dict[mesh_cell_type], dtype = 'int32')

				# Set new points and cells
				if int(float(meshio.__version__.split('.')[0])) >= 4:
					new_mesh_cells_tuple_list = [meshio.CellBlock(cell_type, data) for cell_type, data in new_mesh_cells_dict.items()]	
					meshio_mesh.cells = new_mesh_cells_tuple_list
				else:
					meshio_mesh.cells = new_mesh_cells_dict

				meshio_mesh.points = new_vertices

			elif domain_type == '3D':

				def createNewEntityFromVertices(entity_vertices):
					"""
					Creates a new entity from vertices.
					"""
					return entity_vertices

			else:
				raise ValueError(" ❌ ERROR: domain_type == '%s' is not defined." %(domain_type))

			#################### Export to .xdmf ####################

			utils.customPrint(" 🌀 [meshio] Exporting to .xdmf...")

			# Some converters (like VTK) require `points` to be contiguous.
			points = np.ascontiguousarray(meshio_mesh.points)

			#meshio.write('TESTE-vtk.vtk', meshio_mesh, file_format = 'vtk-ascii')

			# Escrever em arquivo
			XDMF_file_name = '%s/mesh.xdmf' %(problem_folder)

			# If you have HDF5 compiled without zlib support ("I/O filters (external): deflate (zlib)"), it
			 # is strictly necessary to deactivate compression, otherwise there will be an error in FEniCS
			 # (from HDF5) when loading the file.
			 # Check it with: "$ h5pcc -showconfig" or "$ h5cc -showconfig" (whichever is available in your installation).
			 #  https://github.com/tensorflow/io/issues/308
			 # https://forum.hdfgroup.org/t/how-do-i-resolve-this-error/7217/5
			if int(float(meshio.__version__.split('.')[0])) >= 4:
				#meshio.write(XDMF_file_name, meshio_mesh, file_format = 'xdmf', data_format = "XML")
				meshio.write(XDMF_file_name, meshio_mesh, file_format = 'xdmf', compression = None)
			else:
				meshio.write(XDMF_file_name, meshio_mesh, file_format = 'xdmf')

	else:

		# Escrever em arquivo
		XDMF_file_name = '%s/mesh.xdmf' %(problem_folder)

	# Wait for everyone!
	if utils_fenics_mpi.runningInParallel():
		utils_fenics_mpi.waitProcessors()

	###################### Import .xdmf in FEniCS ##########################

	utils.customPrint(" 🌀 Importing from .xdmf to FEniCS Mesh...")

	if use_dolfin_adjoint_if_exists == True:
		try:
			import dolfin_adjoint
			mesh = dolfin_adjoint.Mesh()
		except:
			mesh = Mesh()
	else:
		mesh = Mesh()

	# Read the mesh from the .xdmf file
	mesh_file = XDMFFile(MPI.comm_world, XDMF_file_name)
	mesh_file.read(mesh)
	mesh_file.close()

	########################### Boundary data ##############################

	if return_boundary_data == True:

		# [Parallel]
		if utils_fenics_mpi.runningInParallel():

			# Escrever em arquivo
			XDMF_MeshFunction_file_name = '%s/mesh_function.xdmf' %(problem_folder)

		if utils_fenics_mpi.runningInSerialOrFirstProcessor():

			# [Parallel]
			if utils_fenics_mpi.runningInParallel():

				mesh_parallel = mesh

				mesh_serial = Mesh(MPI.comm_self)

				# Read the mesh from the .xdmf file
				mesh_file_serial = XDMFFile(MPI.comm_self, XDMF_file_name)
				mesh_file_serial.read(mesh_serial)
				mesh_file_serial.close()

				mesh = mesh_serial

			utils.customPrint(" 🌀 Reading boundary data to FEniCS MeshFunction...")

			if domain_type == '2D' or domain_type == '2D axisymmetric':
				mesh.init(1,0) # Initialize connectivity: edge -> vertex
			elif domain_type == '3D':
				mesh.init(2,0) # Initialize connectivity: face -> vertex
			else:
				raise ValueError(" ❌ ERROR: domain_type == '%s' is not defined." %(domain_type))

			# Facets
			mesh_facets_vertices = np.array([facet.entities(0) for facet in facets(mesh)])
			assert len(mesh_facets_vertices) > 0 # Something went wrong...

			# Sort the vertices of each facet in ascending order
			mesh_facets_vertices.sort(axis = 1)

			# Read the OpenFOAM mesh
			foam_reader = FoamReader(problem_folder)
			if utils_fenics_mpi.runningInParallel():
				foam_mesh = foam_reader.readMeshToFoamMesh(mpi_broadcast_result = False, mpi_wait_for_everyone = False)
			else:
				foam_mesh = foam_reader.readMeshToFoamMesh()

			# OpenFOAM vertices of each face
			def sortArray(ar):
				ar.sort(axis = 0) # Sort the vertices of each OpenFOAM mesh face in ascending order
				return ar
			faces_old_vertices = foam_mesh.faces_indices()
			faces_new_vertices = np.array([sortArray(createNewEntityFromVertices(face_vertices)) for face_vertices in faces_old_vertices])

			# OpenFOAM boundaries
			boundaries = foam_mesh.boundaries()

			# Check
			if domain_type == '2D':
				assert 'frontAndBackSurfaces' in boundaries # Sorry, must have it for 2D mesh
			elif domain_type == '2D axisymmetric':
				assert 'frontSurface' in boundaries # Sorry, must have it for 2D axisymmetric mesh
				assert 'backSurface' in boundaries # Sorry, must have it for 2D axisymmetric mesh

			mesh_function_tag_to_boundary_name = {}
			boundaries_info = {}

			# Intialize the marked facets
			marked_facets = [0 for i in range(len(mesh_facets_vertices))]

			marker_number = 1
			for boundary_name in boundaries:

				utils.customPrint(" - Boundary '%s'..." %(boundary_name))

				# Current boundary
				boundary = boundaries[boundary_name]

				if boundary_name not in ['frontAndBackSurfaces', 'frontSurface', 'backSurface']:

					# Face information
					nFaces = boundary['nFaces']
					startFace = boundary['startFace']
					boundary_faces = [startFace + i for i in range(nFaces)]

					# Save marking name
					mesh_function_tag_to_boundary_name[marker_number] = boundary_name

					# Save some boundary information
					boundaries_info[boundary_name] = {}
					boundaries_info[boundary_name]['type'] = boundary['type']
					if 'inGroups' in boundary:
						boundaries_info[boundary_name]['inGroups'] = boundary['inGroups']

					# For all marked faces of the boundary
					for boundary_face in boundary_faces:
						boundary_face_vertices = faces_new_vertices[boundary_face]
						current_facet_index = utils.findIndexOfElementInArray(mesh_facets_vertices, boundary_face_vertices)

						marked_facets[current_facet_index] = marker_number

				marker_number += 1

			# Create the MeshFunction
			mesh_function = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
			#mesh_function.set_all(0)

			# Set all values to the MeshFunction
			mesh_function.set_values(marked_facets)

			# [Parallel]
			if utils_fenics_mpi.runningInParallel():
				mesh = mesh_parallel

				# Write the mesh to .xdmf file
				utils.customPrint(" 🌀 Writing boundary marking to .xdmf...")
				mesh_function_file = XDMFFile(MPI.comm_self, XDMF_MeshFunction_file_name)
				mesh_function_file.write(mesh_function)
				mesh_function_file.close()

		else:

			# Initialize boundary information
			boundaries_info = None

			# Initialize tags
			mesh_function_tag_to_boundary_name = None

		# [Parallel] Broadcast
		if utils_fenics_mpi.runningInParallel():

			# Wait for everyone!
			utils_fenics_mpi.waitProcessors()

			# Load the MeshFunction from file
			mesh_function = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)

			# Read the mesh_function from the .xdmf file
			utils.customPrint(" 🌀 Reading boundary marking from .xdmf...")
			mesh_function_file = XDMFFile(MPI.comm_world, XDMF_MeshFunction_file_name)
			mesh_function_file.read(mesh_function)
			mesh_function_file.close()

			# Boundary information
			boundaries_info = utils_fenics_mpi.broadcastToAll(boundaries_info)

			# Tags
			mesh_function_tag_to_boundary_name = utils_fenics_mpi.broadcastToAll(mesh_function_tag_to_boundary_name)

		# Prepare the boundary data
		boundary_data = {
			'mesh_function' : mesh_function,
			'mesh_function_tag_to_boundary_name' : mesh_function_tag_to_boundary_name,
			'boundaries' : boundaries_info,
		}

		# Reset matching elements search
		utils.resetSearchMatchingElements()

		# [Parallel] Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

		return mesh, boundary_data
	else:

		# Reset matching elements search
		utils.resetSearchMatchingElements()

		# [Parallel] Wait for everyone!
		if utils_fenics_mpi.runningInParallel():
			utils_fenics_mpi.waitProcessors()

		return mesh


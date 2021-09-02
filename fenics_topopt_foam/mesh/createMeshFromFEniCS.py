################################################################################
#                            createMeshFromFEniCS                              #
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

# FoamWriter
from ..io.FoamMeshWriter import FoamMeshWriter

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

########################## createMeshFromFEniCS ################################

def createMeshFromFEniCS(mesh, boundary_data, problem_folder, domain_type = '2D', python_write_precision = 6):
	"""
	Converts a FEniCS mesh to OpenFOAM.
	"""

	# Reset matching elements search
	utils.resetSearchMatchingElements()

	utils.printDebug("\n 🌀 Creating the OpenFOAM mesh from a mesh from FEniCS...")

	# Create the OpenFOAM mesh from FEniCS
	mesh_creator = MeshCreatorFromFEniCStoFoam(mesh, boundary_data, problem_folder, domain_type = domain_type)

	# [Parallel] Wait for everyone!
	if utils_fenics_mpi.runningInParallel():
		utils_fenics_mpi.waitProcessors()

	# Write to file
	mesh_creator.write_to_foam(python_write_precision = python_write_precision)

	# Reset matching elements search
	utils.resetSearchMatchingElements()

	# [Parallel] Wait for everyone!
	if utils_fenics_mpi.runningInParallel():
		utils_fenics_mpi.waitProcessors()

############################### Mesh3DtoFoam ###################################

class Mesh3DtoFoam():
	"""
	3D mesh computing the necessary information required by OpenFOAM.

	-> Optimize ordering -> https://openfoamwiki.net/index.php/RenumberMesh
		-> Performed in the function "createMeshFromFEniCS" of this file.
	-> Specific orderings -> http://openfoamwiki.net/index.php/Write_OpenFOAM_meshes

	"""
	def __init__(self, mesh, domain_type = '3D'):
		self.domain_type = domain_type

		utils.printDebug(" ╎ 🌀 Preparing mesh connectivities...")

		# [Parallel]
		if utils_fenics_mpi.runningInParallel():

			# Save the original mesh
			orig_mesh = mesh

			# Things to prepare beforehand, since they won't be accessible
			 # when computing the rest in a single processor.
			prepare_beforehand_mesh = [
				'geometric_dimension',
				'topology',
				'coordinates',
				'num_cells',
				'num_faces',
				'num_edges',
				'hmin'
			]

			# Unify the mesh
			unified_mesh = utils_fenics_mpi.UnifiedMesh(mesh, 
				prepare_beforehand = prepare_beforehand_mesh,
				proc_destination = 'all'
				)
			self.unified_mesh = unified_mesh

			# Set the unified mesh, for performing the 'init' functions that follows
			mesh = unified_mesh

		# Initialize edge information in the FEniCS mesh (* This way, mesh.num_edges() returns the correct value).
		mesh.init(1)

		# Initialize connectivities
		mesh.init(1,0) # edge-vertex
		mesh.init(2,0) # face-vertex
		mesh.init(2,1) # face-edge
		mesh.init(1,2) # edge-face

		if domain_type == '3D':
			mesh.init(3,0) # cell-vertex
			mesh.init(3,2) # cell-face
			mesh.init(2,3) # face-cell

		# Boundary mesh
		utils.printDebug(" ╎ 🌀 Preparing a boundary mesh...")

		# [Parallel]
		if utils_fenics_mpi.runningInParallel():

			# Change back to the original mesh for creating the BoundaryMesh
			mesh = orig_mesh

		if self.domain_type == '2D' or self.domain_type == '2D axisymmetric':
			bmesh_external = BoundaryMesh(mesh, 
				'exterior', 
				False # False => Don't use FEniCS UFC convention: Use right-oriented facets, because this saves us the effort of having to reorder the edges.
				)

		elif self.domain_type == '3D':
			bmesh_external = BoundaryMesh(mesh, 
				'exterior', 
				True # => Use FEniCS UFC convention, because I guess there is no use having right-oriented facets in this case (i.e., the facets in this case refer to the faces and not the edges, which will have to be ordered).
				)

		# [Parallel]
		if utils_fenics_mpi.runningInParallel():

			# Things to prepare beforehand, since they won't be accessible
			 # when computing the rest in a single processor.
			if self.domain_type == '2D' or self.domain_type == '2D axisymmetric':
				prepare_beforehand_bmesh = [
					'geometric_dimension',
					'topology',
					'coordinates',
					'num_edges',
					['entity_map', 0],
					['entity_map', 1]
				]

			elif self.domain_type == '3D':
				prepare_beforehand_bmesh = [
					'geometric_dimension',
					'topology',
					'coordinates',
					'num_cells',
					'num_faces',
					'num_edges',
					['entity_map', 2]
				]

			# Unify the boundary mesh 
			unified_bmesh_external = utils_fenics_mpi.UnifiedBoundaryMesh(bmesh_external, unified_mesh, 
				prepare_beforehand = prepare_beforehand_bmesh,
				proc_destination = 'all'
				)

			# Set the unified bmesh, for performing the 'init' functions that follows
			bmesh_external = unified_bmesh_external

			# Change back to the unified mesh
			mesh = unified_mesh

		# Initialize connectivities
		if self.domain_type == '2D' or self.domain_type == '2D axisymmetric':
			bmesh_external.init(1) # Compute edge information

		elif self.domain_type == '3D':
			bmesh_external.init(1) # Compute edge information
			bmesh_external.init(1,2) # edge-face

		# [Parallel]
		if utils_fenics_mpi.runningInParallel():
			if utils_fenics_mpi.runningInFirstProcessor():
				with utils_fenics_mpi.first_processor_lock():
					pass # Continuing with only the first processor
			else:
				return # OK. Let's wrap up and then wait

		utils.printDebug(" ╎ 🌀 Starting mesh generation for %s..." %(self.domain_type))

		if self.domain_type == '2D' or self.domain_type == '2D axisymmetric':

			# 2D mesh and vertex coordinates
			mesh_2D = mesh
			mesh_2D_coordinates = mesh_2D.coordinates()

			# Tolerance for the definition of a symmetry axis
			tol_for_symmetry_axis = 1.E-8

			if self.domain_type == '2D':

				# Thickness of the extrusion from 2D to 3D
				 # * The value of the thickness does not really matter (i.e., does not influence the simulation),
				 #    but it is nice to choose a nice thickness for easiness of visualization.
				thickness_of_extrusion = mesh_2D.hmin() #min(2.E-2, mesh_2D.hmin()) # min(1.E-5, mesh_2D.hmin())

				utils.printDebug(" ╎ 🌀 Performing linear extrusion for +%e and -%e..." %(thickness_of_extrusion, thickness_of_extrusion))

			elif self.domain_type == '2D axisymmetric':

				# Maximum radius of the mesh
				max_radius = mesh_2D_coordinates[:,0].max()

				# Maximum radius of the mesh
				min_radius = mesh_2D_coordinates[:,0].min()

				# Determine if you are using a symmetry axis
				using_symmetry_axis = min_radius <= tol_for_symmetry_axis
				self._using_symmetry_axis = using_symmetry_axis

				# According to OpenFOAM, a wedge of small angle (such as of 1°) is recommended to be used.
				 # This seems to be rather important when including the symmetry axis, otherwise OpenFOAM
				 # might return fatal error such as 
					# --> FOAM FATAL ERROR: 
					# wedge backSurface centre plane does not align with a coordinate plane by 0.043073
				# https://cfd.direct/openfoam/user-guide/v6-boundaries/#x24-1730005.2
				# https://stackoverflow.com/questions/51976558/axisymmetric-blockmeshdict-foam-fatal-error-wedge-centre-plane-does-not-al
				desired_angle_degrees = 1.0 # °
				desired_angle = desired_angle_degrees*(2*np.pi)/360. # radians
				thickness_of_extrusion = max_radius*np.tan(desired_angle)

				# Since extremely large radii can result in large thicknesses for the extrusion
				 # and it is not nice to have a large thickness to visualize a result that 
				 # pertains only to a face (* If the thickness is too large, it may make it hard to
				 # visualize in ParaView (* "Imagine" have to maneuver a 1 m thick geometry with a 
				 # face of 10 mm in ParaView)), the maximum thickness is limited by
				 # the minimum diameter of the elements of the mesh (mesh_2D.hmin()).
				if thickness_of_extrusion > mesh_2D.hmin():
					thickness_of_extrusion = mesh_2D.hmin()

				utils.printDebug("""
 ╎ 🌀 Performing rotational extrusion for maximum thicknesses of +%e and -%e...
 ╎ ❗ When writing the mesh to files, if the 'python_write_precision' parameter is not sufficiently high (* For example, a sufficiently high value can be python_write_precision = 10, while a not sufficiently high value can be python_write_precision = 6), OpenFOAM will probably throw various \"not planar\" warnings. This is because that OpenFOAM demands a numerical precision of 10^(-15) (https://bugs.openfoam.org/view.php?id=2126) in order to consider \"planar\" surfaces (* These planar surfaces are the sides of the wedge).
""" %(thickness_of_extrusion, thickness_of_extrusion))
					# https://bugs.openfoam.org/view.php?id=2126
					# https://www.cfd-online.com/Forums/openfoam/176129-solved-wedge-patch-not-planer-error.html
					# https://www.cfd-online.com/Forums/openfoam-meshing/154697-wedge-patch-not-planar.html
					# https://www.cfd-online.com/Forums/openfoam/175893-wedge-patch-not-planer-error-when-pimplefoam.html

			# Thickness of the extrusion from 2D to 3D
			if self.domain_type == '2D':
				# "2D mesh"
				# Using prism elements (https://cfd.direct/openfoam/user-guide/v7-mesh-description/)
				# -> Linear extrusion
				# Coordinates in FEniCS ordered as (x, y, z)
				# In OpenFOAM, x -> x; y -> y; [z] -> z (thickness);

				# •
				# │╲
				# │ ╲
				# │  ╲
				# │   ╲     ---> Becomes a 'prism' element after extrusion to 3D
				# │    ╲
				# │     ╲
				# •──────•

				def getThickness(coords, coord_type = 'first'):
					"""
					Returns the "thickness" dimension of the mesh linearly extruded to 3D.
					"""
					if coord_type == 'first':
						return thickness_of_extrusion/2.0
					elif coord_type == 'second':
						return -thickness_of_extrusion/2.0

			elif self.domain_type == '2D axisymmetric':
				# "2D axisymmetric mesh"
				# Using prism and tetrahedron/pyramid elements (https://cfd.direct/openfoam/user-guide/v7-mesh-description/)
				# Coordinates in FEniCS ordered as (r, z)
				# -> Rotational extrusion
				# In OpenFOAM, for the generation of the mesh, r -> x; z -> y; [θ] -> z (thickness);
				# -> The order of the coordinates is fixed at the end*, 
				#    in order to match the right-hand rule:
				#             3D coordinates:  (x, y, z)
				#                Representing: (r, z, θ) --> Fixing to right-hand rule --> (r, θ, z')
				#                               => In the end, r -> x; [θ] -> y (thickness); z -> z;
				#                * This is just to be able to use the reusing code from self.domain_type == '2D' 
				# Check out: https://openfoamwiki.net/index.php/Main_ContribExamples/AxiSymmetric
				#            https://www.openfoam.com/documentation/guides/latest/doc/guide-bcs-constraint-wedge.html

				# •
				# │╲
				# │ ╲
				# │  ╲
				# │   ╲     ---> Becomes a 'prism' element after rotational extrusion to 3D (because of the "one-element thickness")
				# │    ╲
				# │     ╲
				# •──────•

				#
				# │ <---- Symmetry axis
				# •
				# │╲
				# • ╲
				# │  ╲
				# •   ╲     ---> Becomes a 'tetrahedron' element after rotational extrusion to 3D
				# │    ╲
				# •     ╲
				# │──────•
				# 

				#
				# │ <---- Symmetry axis
				# •
				# │      •
				# •     ╱│
				# │    ╱ │
				# •   ╱  │
				# │  ╱   │  ---> Becomes a 'pyramid' element after rotational extrusion to 3D
				# • ╱    │
				# │╱     │
				# •──────•
				# │
				# 

				def getThickness(coords, coord_type = 'first'):
					"""
					Returns the "thickness" dimension of the mesh rotationally extruded to 3D.
					"""

					# Radial coordinate
					r = coords[0]

					if coord_type == 'first':
						if r <= tol_for_symmetry_axis: # [When assuming symmetry axis, set the coordinate as exactly zero]
							return 0.0
						else: # Wedge
							return r/max_radius*thickness_of_extrusion/2.0

					elif coord_type == 'second':
						if r <= tol_for_symmetry_axis: # [When assuming symmetry axis]
							raise ValueError(" ❌ ERROR: A vertex over the symmetry axis is shared between the two sides of the rotational extrusion and is already considered in coord_type == 'first', but you are trying to duplicate it here!")
						else: # Wedge
							return -r/max_radius*thickness_of_extrusion/2.0

			###################### Coordinates ######################

			utils.printDebug(" ╎ 🌀 Preparing mesh coordinates...")

			def getCorresponding3Dcoordinates(mesh_coordinates, coord_type = 'first'):
				"""
				Returns the corresponding 3D coordinates after extrusion from a 2D mesh.
				"""
				return np.array([[mesh_coordinates[i][0], mesh_coordinates[i][1], getThickness(mesh_coordinates[i], coord_type = coord_type)] for i in range(len(mesh_coordinates))])

			# [Front side] 3D coordinates of the vertices of the front side
			coordinates_frontside = getCorresponding3Dcoordinates(mesh_2D_coordinates, coord_type = 'first')

			# [Back side] 2D coordinates of the vertices of the back side
			 # In the case of "2D axisymmetric", it is needed to remove the vertices that are over the symmetry axis, because only one vertex that is shared between the two faces should be considered over it!
			if self.domain_type == '2D' or (self.domain_type == '2D axisymmetric' and using_symmetry_axis == False):
				mesh_2D_coordinates_for_backside = mesh_2D_coordinates
			elif self.domain_type == '2D axisymmetric':
				indices_on_symmetry_axis = np.where(mesh_2D_coordinates[:,0] <= tol_for_symmetry_axis)[0]
				mesh_2D_coordinates_for_backside = np.delete(mesh_2D_coordinates, indices_on_symmetry_axis, axis = 0)

				# Crop r <= tol_for_symmetry_axis to zero
				coordinates_frontside[indices_on_symmetry_axis, 0] = 0.0

			coordinates_backside = getCorresponding3Dcoordinates(mesh_2D_coordinates_for_backside, coord_type = 'second')

			# Save the vertex coordinates
			self._coordinates = np.append(coordinates_frontside, coordinates_backside, axis = 0)

			# Save the number of vertices
			self.num_vertices = len(self._coordinates)

			################### External faces ######################

			##################### Front side ########################

			utils.printDebug(" ╎ 🌀 Preparing external faces (front side)...")

			# Get all faces of mesh_2D
			 # * Entities "0" refer to the vertices of the mesh.
			 # * mesh_2D_faces contains a NumPy array with the indices of the vertices
			mesh_2D_faces = np.array([face.entities(0) for face in utils_fenics_mpi.get_faces(mesh_2D)])

			def prepare_and_order_external_face_on_the_frontside(face_indices):
				"""
				Prepares and orders the vertices of an external face in the front side.
				* The vertices of a face MUST be ordered with the right-hand 
				   rule, in which the face normal vector MUST be pointing to the outside of the domain.
				   https://cfd.direct/openfoam/user-guide/v7-mesh-description/
				"""

				# Coordinates of the vertices of the face
				face_coords = [mesh_2D_coordinates[i_vertex] for i_vertex in face_indices]

				# Normal vector to face
				normal_to_face = utils_fenics.computeNormalVector(face_coords, normalize = True)

				z_coord = normal_to_face[2]
				if z_coord < 0:
					return utils.revertOrder(face_indices)
				else:
					return face_indices

			# External faces of the front side
			external_faces_frontside = np.array([prepare_and_order_external_face_on_the_frontside(face2D) for face2D in mesh_2D_faces])

			###################### Back side ########################

			utils.printDebug(" ╎ 🌀 Preparing external faces (back side)...")

			# Data type for the indices: 'uint32' or 'uint64'
			indices_dtype = mesh.cells().dtype.name

			# Number of vertices in the front side
			num_coordinates_frontside = len(coordinates_frontside)

			# Create a map from the index of a vertex of the front side to 
			 # the index of the corresponding vertex on the back side.
			if self.domain_type == '2D' or (self.domain_type == '2D axisymmetric' and using_symmetry_axis == False):
				map_vertex_frontside_to_backside = np.array([i + num_coordinates_frontside for i in range(num_coordinates_frontside)], dtype = indices_dtype)
					# * Can't sum NumPy unsigned number (uint32, uint64, ulonglong ...) with Python int,
					 # because it returns float! https://github.com/numpy/numpy/issues/3118
					 # Therefore, the type 'uint32' (or 'uint64') needs to be explicit!
			elif self.domain_type == '2D axisymmetric':

				# Tag for a non-existent vertex ID
				nonexistent_vertex_id = utils.get_max_unsigned_int_c()

				# Number of times a vertex on the symmetry axis has been skipped in the mapping
				jump_count = 0 

				map_vertex_frontside_to_backside_list = []
				for i_frontside in range(num_coordinates_frontside):
					if i_frontside in indices_on_symmetry_axis:
						map_vertex_frontside_to_backside_list += [i_frontside]
						#map_vertex_frontside_to_backside_list += [nonexistent_vertex_id]
						jump_count += 1
					else:
						map_vertex_frontside_to_backside_list += [i_frontside + num_coordinates_frontside - jump_count]

				map_vertex_frontside_to_backside = np.array(map_vertex_frontside_to_backside_list, dtype = indices_dtype)
					# * Can't sum NumPy unsigned number (uint32, uint64, ulonglong ...) with Python int,
					 # because it returns float! https://github.com/numpy/numpy/issues/3118
					 # Therefore, the type 'uint32' (or 'uint64') needs to be explicit!

			def map_face_frontside_to_backside(mesh_face):
				"""
				Revert the order of the vertex indices of a face.
				* i.e.,
					clockwise -> counterclockwise
					counterclockwise -> clockwise
				"""
				return utils.revertOrder([map_vertex_frontside_to_backside[mesh_face[j]] for j in range(len(mesh_face))])

			def map_faces_frontside_to_backside(mesh_faces):
				"""
				Map faces from the front side to the back side.
				* The list of vertices of a face is reverted,
				  in order for the normal vector (from the 
				  right-hand-rule) to point outside.
				"""
				return np.array([map_face_frontside_to_backside(mesh_faces[i]) for i in range(len(mesh_faces))])

			# External faces of the back side
			external_faces_backside = map_faces_frontside_to_backside(external_faces_frontside)

			#################### Thickness side #####################

			utils.printDebug(" ╎ 🌀 Preparing external faces (thickness side)...")
			vertex_map_from_bmesh_to_mesh = utils_fenics.mapBoundaryMeshToMesh(bmesh_external, entity_type = 'vertex')
			edge_map_from_bmesh_to_mesh = utils_fenics.mapBoundaryMeshToMesh(bmesh_external, entity_type = 'edge')

			def convert_bmesh_vertices_to_mesh_vertices(bmesh_vertices):
				"""
				Convert the indices of boundary mesh vertices to mesh indices. 
				"""
				return np.array([vertex_map_from_bmesh_to_mesh[bmesh_vertices[i]] for i in range(len(bmesh_vertices))])

			# External edges of the 2D mesh
			mesh_2D_external_edges_vertices = np.array([convert_bmesh_vertices_to_mesh_vertices(edge.entities(0)) for edge in utils_fenics_mpi.get_edges(bmesh_external)])
			mesh_2D_external_edges_ids = edge_map_from_bmesh_to_mesh # Obs. It is a list of edges of mesh_2D

			# Boundary edges of the final 3D mesh for the front side
			boundary_edges_frontside = mesh_2D_external_edges_vertices

			# Boundary edges of the final 3D mesh for the back side
			boundary_edges_backside = map_faces_frontside_to_backside(boundary_edges_frontside)

			def prepare_and_order_external_face_on_the_thickness(frontside_vertices, backside_vertices):
				"""
				Prepares and orders an external face in the 'thickness' direction of the 3D mesh.
				* The vertices of a face MUST be ordered with the right-hand 
				   rule, in which the face normal vector MUST be pointing to the outside of the domain.
				   https://cfd.direct/openfoam/user-guide/v7-mesh-description/
				"""

				# Array of all vertices => Prism element (6 vertices)
				face_vertices = utils.revertOrder(np.append(frontside_vertices, backside_vertices, axis = 0))

				# If the symmetry axis is present => Tetrahedron element (4 vertices) or pyramid element (5 vertices)
				if self.domain_type == '2D' or (self.domain_type == '2D axisymmetric' and using_symmetry_axis == False):
					pass
				elif self.domain_type == '2D axisymmetric':
					if np.any(np.isin(face_vertices, indices_on_symmetry_axis)):
						face_vertices = np.array(utils.removeRepeatedElementsFromList(face_vertices.tolist(), keep_order = True))
				return face_vertices

			def map_frontside_and_backside_boundary_edges_to_faces(boundary_edges_frontside, boundary_edges_backside):
				"""
				Map the front side and back side boundary edges to faces on the 'thickness' side.
				"""

				if self.domain_type == '2D' or (self.domain_type == '2D axisymmetric' and using_symmetry_axis == False):

					# Create the faces of the "thickness" side of the 3D mesh
					external_faces_thickness_side = np.array([prepare_and_order_external_face_on_the_thickness(boundary_edges_frontside[i], boundary_edges_backside[i]) for i in range(len(boundary_edges_frontside))])

					# Map the boundary edges' indices to the faces of the "thickness" side of the 3D mesh (* In this case, the correspondence is 1:1)
					map_boundary_edges_to_thickness_side = {
						edge_map_from_bmesh_to_mesh[i] : 
							len(external_faces_frontside) + len(external_faces_backside) + i
								for i in range(len(boundary_edges_frontside))
						}

				elif self.domain_type == '2D axisymmetric':
					external_faces_thickness_side = []
					map_boundary_edges_to_thickness_side = {}

					# For all boundary edges of the front side
					for i in range(len(boundary_edges_frontside)):

						if np.all(boundary_edges_frontside[i] == utils.revertOrder(boundary_edges_backside[i])): # If the boundary edge is on the symmetry axis
							# This is NOT a face, because it collapsed into the symmetry axis itself!
							map_boundary_edges_to_thickness_side[edge_map_from_bmesh_to_mesh[i]] = None
						else:
							map_boundary_edges_to_thickness_side[edge_map_from_bmesh_to_mesh[i]] = len(external_faces_frontside) + len(external_faces_backside) + len(external_faces_thickness_side)
							external_faces_thickness_side += [prepare_and_order_external_face_on_the_thickness(boundary_edges_frontside[i], boundary_edges_backside[i])]

					# Convert to NumPy array
					external_faces_thickness_side = np.array(external_faces_thickness_side)

				return external_faces_thickness_side, map_boundary_edges_to_thickness_side

			# Map the front side and back side boundary edges to faces on the 'thickness' side. 
			(external_faces_thickness_side, map_boundary_edges_to_thickness_side) = map_frontside_and_backside_boundary_edges_to_faces(boundary_edges_frontside, boundary_edges_backside)
			self._map_boundary_edges_to_thickness_side = map_boundary_edges_to_thickness_side

			############### Set the external faces ##################

			utils.printDebug(" ╎ 🌀 Setting up external faces...")

			external_faces = utils.numpy_append_with_multiple_dimensions(external_faces_frontside, external_faces_backside, external_faces_thickness_side, axis = 0)

			# Save the external faces
			self._external_faces = external_faces
			self.num_external_faces_for_boundary_condition = len(external_faces_thickness_side)
			self.num_external_faces_frontside = len(external_faces_frontside)
			self.num_external_faces_backside = len(external_faces_backside)
			self.num_external_faces_frontside_backside = len(external_faces_frontside) + len(external_faces_backside)

			######################## Cells ##########################

			utils.printDebug(" ╎ 🌀 Preparing cells, internal faces, internal face neighbours and external face neighbours...")

			# Edges
			mesh_2D_edges_vertices = np.array([edge.entities(0) for edge in utils_fenics_mpi.get_edges(mesh_2D)])

			# Internal edge IDs of the 2D mesh
			#mesh_2D_all_edge_ids = np.array([i for i in range(mesh_2D.num_edges())])
			#mesh_2D_internal_edges_ids = np.delete(mesh_2D_all_edge_ids, mesh_2D_external_edges_ids, axis = 0)

			# Edges of the 2D mesh
			mesh_2D_edges_entities = list(utils_fenics_mpi.get_edges(mesh_2D))
			mesh_2D_face_edges = [[mesh_2D_edges_entities[edge_id].entities(0) for edge_id in face.entities(1)] for face in utils_fenics_mpi.get_faces(mesh_2D)]
			mesh_2D_face_edges_ids = [face.entities(1) for face in utils_fenics_mpi.get_faces(mesh_2D)]

			# Boundary faces
			mesh_2D_faces_from_external_edge = [mesh_2D_edges_entities[edge_id].entities(2) for edge_id in mesh_2D_external_edges_ids]
			mesh_2D_faces_to_external_edges = {}
			for i_edge in range(len(mesh_2D_faces_from_external_edge)):
				current_faces = mesh_2D_faces_from_external_edge[i_edge]
				for id_face in current_faces:
					if id_face in mesh_2D_faces_to_external_edges:
						mesh_2D_faces_to_external_edges[id_face] += [i_edge]
					else:
						mesh_2D_faces_to_external_edges[id_face] = [i_edge]

			# Create size counts
			self.num_cells = mesh_2D.num_faces()
			self.num_internal_faces = mesh_2D.num_edges() - bmesh_external.num_edges()
			self.num_external_faces = len(external_faces)
			self.num_faces = self.num_internal_faces + self.num_external_faces

			# Create lists
			cells = []
			internal_faces = [] 
			external_face_neighbours = [[] for i in range(self.num_external_faces)]
			internal_face_neighbours = [[] for i in range(self.num_internal_faces)]

			def getNewAndOldEdges(edges_of_face, edges_already_ordered, method = 'matching indices'):
				"""
				Returns the edges of 'edges_of_faces' that are not already included in 'edges_already_ordered'
				"""

				if method == 'python list':
					new_edges = []
					old_edges = []
					for edge_of_face in edges_of_face:
						if edge_of_face in edges_already_ordered:
							old_edges += [edge_of_face]
						else:
							new_edges += [edge_of_face]

				elif method == 'matching indices':
					old_edges = utils.searchMatchingElements(edges_of_face, edges_already_ordered, profile_key = 'getNewAndOldEdges')
					new_edges = edges_of_face.copy().tolist()

					for old_edge in old_edges:
						new_edges.remove(old_edge)

				else:
					raise ValueError(" ❌ ERROR: method == '%s' is not defined!" %(method))

				return new_edges, old_edges

			def convertVertexIndicesToCoordinates(array_of_indices):
				"""
				Convert the indices of the vertices to coordinates.
				"""
				return np.array([self._coordinates[array_of_indices[i]] for i in range(len(array_of_indices))])

			def orderFacesInCellToOutside(faces_to_order, faces_already_ordered):
				"""
				Order all the 'faces_to_order' in order for their 
				normal vectors to point outside. Since the cells are created in order (0, 1, 2...),
				all of the faces will contain normals pointing to the outside of the lowest numbered cells 
				towards higher numbered cells.

				* This is done in order to comply with the requirement:
				"For each internal face, the ordering of the point labels is such that the face normal points into the cell with the larger label, i.e.  for cells 2 and 5, the normal points into 5; "
					https://cfd.direct/openfoam/user-guide/v7-mesh-description/
				"""

				# Convert indices to coordinates
				coords_faces_to_order = [convertVertexIndicesToCoordinates(faces_to_order[i]) for i in range(len(faces_to_order))]
				coords_faces_already_ordered = [convertVertexIndicesToCoordinates(faces_already_ordered[i]) for i in range(len(faces_already_ordered))]

				# Get all center coordinates
				centers_to_order = [utils.computeCenterCoordinates(coords_faces_to_order[i]) for i in range(len(coords_faces_to_order))]
				centers_already_ordered = [utils.computeCenterCoordinates(coords_faces_already_ordered[i]) for i in range(len(coords_faces_already_ordered))]

				# https://stackoverflow.com/questions/57434507/determing-the-direction-of-face-normals-consistently

				# Get the cell center
				cell_center_coordinates = utils.computeCenterCoordinates(np.array(centers_to_order + centers_already_ordered))

				current_face_directions = []
				for i_face in range(len(coords_faces_to_order)):

					# Coordinates of the vertices of the face
					face_coords = coords_faces_to_order[i_face]

					# Normal vector to face
					normal_to_face = utils_fenics.computeNormalVector(face_coords, normalize = True)

					# Vector from cell center to face
					current_center_coord = centers_to_order[i_face]
					vector_from_cell_center_to_face = current_center_coord - cell_center_coordinates

					# Verdict
					current_face_directions += ['outside' if np.dot(normal_to_face, vector_from_cell_center_to_face) >= 0.0 else 'inside']

				# Adjust for all faces to point outside!
				new_ordered_faces = []
				for i_face in range(len(current_face_directions)):
					if current_face_directions[i_face] == 'inside':
						new_ordered_faces += [utils.revertOrder(faces_to_order[i_face])]
					else:
						new_ordered_faces += [faces_to_order[i_face]]

				return new_ordered_faces

			# IDs of the external edges of the 2D mesh, which are to be included in 'external_face_neighbours'!
			ids_external_edges_mesh_2D_to_include = [mesh_2D_external_edges_ids[i] for i in range(len(mesh_2D_external_edges_ids))]

			# IDs of the edges of the 2D mesh which correspond to already ordered faces.
			ids_edges_mesh_2D_already_ordered = [mesh_2D_external_edges_ids[i] for i in range(len(mesh_2D_external_edges_ids))]

			# Map for obtaining the internal faces from the edges of the 2D mesh
			map_2D_edge_to_internal_face = {}

			# Print progress
			utils.printProgress(0, max_iterations = len(mesh_2D_face_edges_ids))

			# For all faces of the 2D mesh, because each face of the 2D mesh corresponds to a cell in the extruded 3D mesh
			for i_face2D in range(len(mesh_2D_face_edges_ids)):

				# Current face of the 2D mesh being analyzed
				face2D_edges_ids = mesh_2D_face_edges_ids[i_face2D]

				# Get the new and the old edge IDs
				(new_edge_ids, old_edge_ids) = getNewAndOldEdges(face2D_edges_ids, ids_edges_mesh_2D_already_ordered)

				# Get the new and the old edge vertices
				all_mesh_2D_face_edges = mesh_2D_face_edges[i_face2D]
				new_mesh_2D_face_edges = [mesh_2D_edges_vertices[i_new_edge] for i_new_edge in new_edge_ids]
				old_mesh_2D_face_edges = [mesh_2D_edges_vertices[i_old_edge] for i_old_edge in old_edge_ids]

				# Increment the edges that are already ordered
				ids_edges_mesh_2D_already_ordered += new_edge_ids

				# Create the new internal faces from each edge of 'new_mesh_2D_face_edges'
				faces_to_order = []
				for new_mesh_2D_edge in new_mesh_2D_face_edges:
					frontside_edge_vertices = new_mesh_2D_edge
					backside_edge_vertices = map_face_frontside_to_backside(frontside_edge_vertices)

					if self.domain_type == '2D' or (self.domain_type == '2D axisymmetric' and using_symmetry_axis == False):

						faces_to_order += [np.append(frontside_edge_vertices, backside_edge_vertices)]

					elif self.domain_type == '2D axisymmetric':

						new_face = np.append(frontside_edge_vertices, backside_edge_vertices)

						# Remove repeated elements
						new_face = np.array(utils.removeRepeatedElementsFromList(new_face.tolist(), keep_order = True))

						if len(new_face) > 2: # i.e., if it is not the symmetry axis
							faces_to_order += [new_face]

				# Create the already ordered internal faces
				faces_already_ordered = []
				for old_mesh_2D_edge in old_mesh_2D_face_edges:
					frontside_edge_vertices = old_mesh_2D_edge
					backside_edge_vertices = map_face_frontside_to_backside(frontside_edge_vertices)

					if self.domain_type == '2D' or (self.domain_type == '2D axisymmetric' and using_symmetry_axis == False):

						faces_already_ordered += [np.append(frontside_edge_vertices, backside_edge_vertices)]

					elif self.domain_type == '2D axisymmetric':

						new_face = np.append(frontside_edge_vertices, backside_edge_vertices)

						# Remove repeated elements
						new_face = np.array(utils.removeRepeatedElementsFromList(new_face.tolist(), keep_order = True))

						if len(new_face) > 2: # i.e., if it is not the symmetry axis
							faces_already_ordered += [new_face]

				# Order the faces in order for the normals of the 'faces_to_order' to point correctly in OpenFOAM
				new_ordered_faces = orderFacesInCellToOutside(faces_to_order, faces_already_ordered)

				###################### cells ####################
				# 1 cell created per face of the 2D mesh

				# Save the coordinates of the vertices in cells

				# Get all vertices of the face of the 2D mesh	
				vertices_face2D = []
				for edge_of_face in all_mesh_2D_face_edges:
					vertices_face2D += edge_of_face.tolist()

				# Remove repeated elements by converting to set and going back to list/array
				vertices_face2D = np.array(list(set(vertices_face2D))) 

				# Front side vertices
				frontside_cell_vertices = vertices_face2D

				# Back side vertices
				backside_cell_vertices = map_face_frontside_to_backside(vertices_face2D)

				# Include the cell
				cells += [np.append(frontside_cell_vertices, backside_cell_vertices)]

				# Current cell number
				current_cell_number = len(cells) - 1

				############ internal_face_neighbours ###########

				# Update internal_face_neighbours with the current cell number

				# For the new faces created for the cell
				numbers_of_the_new_faces = [len(internal_faces) + i for i in range(len(new_ordered_faces))]
				for number_of_new_face in numbers_of_the_new_faces:
					internal_face_neighbours[number_of_new_face] += [current_cell_number]

				# For the old faces that were already available for the cell
				for i_edge in range(len(old_edge_ids)):
					old_edge_id = old_edge_ids[i_edge]

					if utils.checkIfInArray(old_edge_id, ids_external_edges_mesh_2D_to_include, profile_key = 'edge_search') == False:
					#if old_edge_id not in array_with_findable_elements:
						number_of_old_face = map_2D_edge_to_internal_face[old_edge_id]
						internal_face_neighbours[number_of_old_face] += [current_cell_number]

				# Update map for obtaining the internal faces from the edges of the 2D mesh
				for i_edge in range(len(new_edge_ids)):
					map_2D_edge_to_internal_face[new_edge_ids[i_edge]] = numbers_of_the_new_faces[i_edge]

				################## internal_faces ###############

				# Include the new faces as internal_faces
				internal_faces += new_ordered_faces

				############ external_face_neighbours ###########

				#### Front side external faces
				external_faces_frontside_to_include = [i_face2D]

				#### Back side external faces
				external_faces_backside_to_include = [i_face2D + len(external_faces_frontside)]

				#### Thickness side external faces

				# Check the external edges that are already ordered, because we need to include them.
				if self.domain_type == '2D' or (self.domain_type == '2D axisymmetric' and using_symmetry_axis == False):
					face2D_external_edges = mesh_2D_faces_to_external_edges.get(i_face2D, [])
					external_faces_thickness_side_to_include = [face2D_external_edge + len(external_faces_frontside) + len(external_faces_backside) for face2D_external_edge in face2D_external_edges]
				elif self.domain_type == '2D axisymmetric':
					face2D_external_edges = mesh_2D_faces_to_external_edges.get(i_face2D, [])

					face2D_external_edges_thickness_side = [map_boundary_edges_to_thickness_side[edge_map_from_bmesh_to_mesh[face2D_external_edge]] for face2D_external_edge in face2D_external_edges]
					external_faces_thickness_side_to_include = [face2D_external_edge for face2D_external_edge in face2D_external_edges_thickness_side]

					if None in external_faces_thickness_side_to_include:
						external_faces_thickness_side_to_include.remove(None)

				#### All external faces
				external_faces_to_include = external_faces_frontside_to_include + external_faces_backside_to_include + external_faces_thickness_side_to_include

				# Update external_face_neighbours
				 # * Remember that external faces have only 1 neighbor cell
				for i_external_face in external_faces_to_include:
					external_face_neighbours[i_external_face] += [current_cell_number]

				# Print progress
				utils.printProgress(i_face2D + 1)

			#########################################################

			utils.printDebug(" ╎ 🌀 Setting up cells, internal faces, internal face neighbours and external face neighbours...")

			####################### cells ###########################

			self._cells = cells

			################### internal_faces ######################

			self._internal_faces = internal_faces

			################# internal_face_neighbours ##############

			# Internal faces have two cell neighbours
			 # Ordering of external faces MUST follow the mesh_function marking
			self._internal_face_neighbours = np.array(internal_face_neighbours)

			################# external_face_neighbours ##############

			# External faces have one cell neighbour
			 # Ordering of external faces MUST follow the mesh_function marking
			self._external_face_neighbours = np.array(external_face_neighbours)

			##################### Fix coordinates ###################

			if self.domain_type == '2D axisymmetric':
				# Fix the order of the coordinates of the 2D axisymmetric mesh
				# 3D coordinates:  (x, y, z)
				#    Representing: (r, z', θ) --> Fixing* to right-hand rule --> (r, θ, z')
				#    => r = x, θ = -z, z = y 
				#     * By swapping columns (using advanced slicing --- https://stackoverflow.com/questions/4857927/swapping-columns-in-a-numpy-array)
				self._coordinates = self._coordinates[:,[0, 2, 1]]
				self._coordinates[:,1] *= (-1)

		elif self.domain_type == '3D':

			self._mesh = mesh

			###################### Coordinates ######################

			utils.printDebug(" ╎ 🌀 Preparing mesh coordinates...")

			coordinates = mesh.coordinates()
			self._coordinates = coordinates
			self.num_vertices = len(self._coordinates)

			######################## Cells ##########################

			utils.printDebug(" ╎ 🌀 Preparing cells, external faces, internal faces, internal face neighbours and external face neighbours...")

			# Map from bmesh to mesh
			face_map_from_bmesh_to_mesh = utils_fenics.mapBoundaryMeshToMesh(bmesh_external, entity_type = 'face')

			# Boundary faces of the mesh
			mesh_boundary_faces = face_map_from_bmesh_to_mesh

			# Necessary mesh information
			mesh_cells_vertices = np.array([cell.entities(0) for cell in utils_fenics_mpi.get_cells(mesh)])
			mesh_cells_faces = np.array([cell.entities(2) for cell in utils_fenics_mpi.get_cells(mesh)])

			mesh_faces_vertices = np.array([face.entities(0) for face in utils_fenics_mpi.get_faces(mesh)])
			mesh_faces_cells = np.array([face.entities(3) for face in utils_fenics_mpi.get_faces(mesh)])
				# , dtype = 'object'

			def orderFaceInCell(current_face_vertices, cell_vertices, direction = 'outside'):
				"""
				Order all vertices of 'current_face_vertices' in order for their 
				normal vectors to point outside/inside the cell. Since the cells are created in order (0, 1, 2...),
				all of the faces will contain normals pointing to the outside of the lowest numbered cells 
				towards higher numbered cells.

				* This is done in order to comply with the requirement:
				"For each internal face, the ordering of the point labels is such that the face normal points into the cell with the larger label, i.e.  for cells 2 and 5, the normal points into 5; "
					https://cfd.direct/openfoam/user-guide/v7-mesh-description/
				"""

				# Convert indices to coordinates
				face_coords = np.array([coordinates[current_face_vertices[i]] for i in range(len(current_face_vertices))])
				coords_cell_vertices = np.array([coordinates[cell_vertices[i]] for i in range(len(cell_vertices))])

				# Compute the center coordinates
				face_center_coords = utils.computeCenterCoordinates(face_coords)
				cell_center_coords = utils.computeCenterCoordinates(coords_cell_vertices)

				# https://stackoverflow.com/questions/57434507/determing-the-direction-of-face-normals-consistently

				# Normal vector to face
				normal_to_face = utils_fenics.computeNormalVector(face_coords, normalize = True)

				# Vector from cell center to face
				vector_from_cell_center_to_face = face_center_coords - cell_center_coords

				if direction == 'outside':
					if np.dot(normal_to_face, vector_from_cell_center_to_face) >= 0.0: # Pointing outside
						return current_face_vertices
					else: # Pointing inside
						return utils.revertOrder(current_face_vertices)
				elif direction == 'inside':
					if np.dot(normal_to_face, vector_from_cell_center_to_face) >= 0.0: # Pointing outside
						return utils.revertOrder(current_face_vertices)
					else: # Pointing inside
						return current_face_vertices
				else:
					raise ValueError(" ❌ ERROR: direction == '%s' is not defined!" %(method))


			cells = []
			map_from_mesh_cell_index_to_cell_index = {}
			map_from_mesh_face_index_to_external_face_index = {}

			external_faces = [] 
			internal_faces = [] 
			external_face_neighbours = []
			internal_face_neighbours = []

			# Print progress
			utils.printProgress(0, max_iterations = len(mesh_faces_vertices))

			# For all faces
			for i_mesh_face in range(len(mesh_faces_vertices)):

				# Vertices of the face
				face_vertices = mesh_faces_vertices[i_mesh_face]

				# Cells of the face
				face_cells = mesh_faces_cells[i_mesh_face]

				if len(face_cells) == 1: # External face (only one neighbour)

					# Index of the only cell neighbour in the mesh
					mesh_cell_index = face_cells[0]

					# Vertices of the only cell neighbour
					cell_vertices = mesh_cells_vertices[mesh_cell_index]

					# Reorder the face for the normal to the new external face to point outside of the mesh
					new_face = orderFaceInCell(face_vertices, cell_vertices, direction = 'outside')

					# Include the new external face
					external_faces += [new_face]
					map_from_mesh_face_index_to_external_face_index[i_mesh_face] = [len(external_faces) - 1]

					# Include the new cell if it has not been included yet
					if mesh_cell_index not in map_from_mesh_cell_index_to_cell_index:
						cells += [cell_vertices]
						current_cell_number = len(cells) - 1
						map_from_mesh_cell_index_to_cell_index[mesh_cell_index] = current_cell_number
					else:
						current_cell_number = map_from_mesh_cell_index_to_cell_index[mesh_cell_index]

					# Include the current cell in the neighbours of the new external face
					external_face_neighbours += [current_cell_number]

				else: # Internal face (two neighbours). I think we don't need an 'assert len(face_cells) == 2' here, because this number is always to equal to 2 if the mesh exists.
				
					# Indices of the cells neighbours in the mesh
					mesh_cell0_index = face_cells[0]
					mesh_cell1_index = face_cells[1]

					# Vertices of the cells neighbours in the mesh
					cell0_vertices = mesh_cells_vertices[mesh_cell0_index]
					cell1_vertices = mesh_cells_vertices[mesh_cell1_index]

					# Get the cell numbers
					 # Check if any of the cells has already been included
					if mesh_cell0_index not in map_from_mesh_cell_index_to_cell_index:
						cells += [cell0_vertices]
						cell0_number = len(cells) - 1
						map_from_mesh_cell_index_to_cell_index[mesh_cell0_index] = cell0_number
					else:
						cell0_number = map_from_mesh_cell_index_to_cell_index[mesh_cell0_index]

					if mesh_cell1_index not in map_from_mesh_cell_index_to_cell_index:
						cells += [cell1_vertices]
						cell1_number = len(cells) - 1
						map_from_mesh_cell_index_to_cell_index[mesh_cell1_index] = cell1_number
					else:
						cell1_number = map_from_mesh_cell_index_to_cell_index[mesh_cell1_index]

					# Include the current cells in the neighbours of the new internal face
					ordered_cell_numbers = [cell0_number, cell1_number]
					ordered_cell_numbers.sort(reverse = False)
					internal_face_neighbours += [ordered_cell_numbers]

					#### Reorder the face

					# Lowest numbered cell
					lowest_numbered_cell = ordered_cell_numbers[0]

					# Vertices of the lowest numbered cell
					if ordered_cell_numbers.index(cell0_number) == 0:
						lowest_numbered_cell_vertices = cell0_vertices
					else: # ordered_cell_numbers.index(cell1_number) == 0:
						lowest_numbered_cell_vertices = cell1_vertices
					
					# Reorder the face for the normal to the new internal face to point outside of the lowest numbered cell
					new_face = orderFaceInCell(face_vertices, lowest_numbered_cell_vertices, direction = 'outside')

					# Include the new internal face
					internal_faces += [new_face]

				# Print progress
				utils.printProgress(i_mesh_face + 1)

			# Save the face map of the external faces
			self._map_from_mesh_face_index_to_external_face_index = map_from_mesh_face_index_to_external_face_index

			####################### cells ###########################

			self._cells = np.array(cells)

			################### external_faces ######################

			self._external_faces = np.array(external_faces)

			################### internal_faces ######################

			self._internal_faces = np.array(internal_faces)

			################# internal_face_neighbours ##############

			# Internal faces have two cell neighbours
			 # Ordering of external faces MUST follow the mesh_function marking
			self._internal_face_neighbours = np.array(internal_face_neighbours)

			################# external_face_neighbours ##############

			# External faces have one cell neighbour
			 # Ordering of external faces MUST follow the mesh_function marking
			self._external_face_neighbours = np.array(external_face_neighbours)

			# Create size counts
			self.num_cells = len(self._cells)
			self.num_internal_faces = len(self._internal_faces)
			self.num_external_faces = len(self._external_faces)
			self.num_faces = self.num_internal_faces + self.num_external_faces

	########################## Boundary information ########################

	def setBoundaryInformation(self, boundary_data, boundaries_to_consider = 'external'):
		"""
		Sets the necessary boundary information.
		Attention: The order of the faces is changed with this operation!
		                                     -------

		Check out:
			https://www.openfoam.com/documentation/user-guide/boundaries.php#x12-390004.2.1
		"""

		utils.customPrint(" ╎ 🌀 Setting up boundary information...")

		self._boundary_information = {}

		if boundaries_to_consider != 'external':
			raise ValueError(" ❌ ERROR: Cannot consider '%s' boundaries (at least yet)!" %(boundaries_to_consider))

		mesh_function = boundary_data['mesh_function']
		mesh_function_tag_to_boundary_name = boundary_data['mesh_function_tag_to_boundary_name']

		if self.domain_type == '2D' or self.domain_type == '2D axisymmetric':

			#### External faces of the front side and back side #####

			# Add the necessary boundary conditions that enable 2D or 2D axisymmetric simulation
			if self.domain_type == '2D':

				self._boundary_information['frontAndBackSurfaces'] = {
					'type' : 'empty',
					'inGroups' : ['empty'],
					'nFaces' : self.num_external_faces_frontside_backside,
					'startFace' : self.num_internal_faces + 0,
				}

			elif self.domain_type == '2D axisymmetric':

				# Why two separated surfaces?
					# https://bugs.openfoam.org/view.php?id=1188
					# https://openfoamwiki.net/index.php/Main_ContribExamples/AxiSymmetric

				self._boundary_information['frontSurface'] = {
					'type' : 'wedge',
					'inGroups' : ['wedge'],
					'nFaces' : self.num_external_faces_frontside,
					'startFace' : self.num_internal_faces + 0,
				}

				self._boundary_information['backSurface'] = {
					'type' : 'wedge',
					'inGroups' : ['wedge'],
					'nFaces' : self.num_external_faces_backside,
					'startFace' : self.num_internal_faces + self.num_external_faces_frontside + 0,
				}

			############ Set the other boundary conditions ##########

			new_indices_external_faces_thickness_side = []
			for mesh_function_tag in mesh_function_tag_to_boundary_name:

				# startFace of the boundary
				current_boundary_position = len(new_indices_external_faces_thickness_side)

				# Name of the boundary
				boundary_name = mesh_function_tag_to_boundary_name[mesh_function_tag]

				# Reserved names
				assert boundary_name not in ['frontAndBackSurfaces', 'frontSurface', 'backSurface']

				utils.customPrint(" ╎    - Boundary: '%s'..." %(boundary_name))

				if boundary_name == 'unset': # It's assumed to be a dummy marker. Please, don't leave anything marked in it!
					utils.customPrint(" ╎    -> Skipped ('%s' assumed as a dummy marker)..." %(boundary_name))
				else:

					# Find the indices of the boundary
					edges_of_the_boundary = mesh_function.where_equal(mesh_function_tag)
					indices_faces_of_the_boundary = [self._map_boundary_edges_to_thickness_side[edges_of_the_boundary[i]] for i in range(len(edges_of_the_boundary))]
						# * Boundary condition is not appliable on the symmetry axis, because this boundary is ignored. Nonetheless, it is not needed, because of the "special" boundary condition considered.

					if self.domain_type == '2D' or (self.domain_type == '2D axisymmetric' and self._using_symmetry_axis == False):
						pass
					elif self.domain_type == '2D axisymmetric':
						while None in indices_faces_of_the_boundary:
							indices_faces_of_the_boundary.remove(None)

					#edge_map_from_bmesh_to_mesh

					# Include the boundary information specified by the user
					self._boundary_information[boundary_name] = {}
					assert 'type' in boundary_data['boundaries'][boundary_name]

					for key in boundary_data['boundaries'][boundary_name]:
						self._boundary_information[boundary_name][key] = boundary_data['boundaries'][boundary_name][key]

					# Include the computed boundary location
					self._boundary_information[boundary_name]['nFaces'] = len(indices_faces_of_the_boundary)
					self._boundary_information[boundary_name]['startFace'] = self.num_internal_faces + self.num_external_faces_frontside_backside + current_boundary_position

					# Update
					new_indices_external_faces_thickness_side += indices_faces_of_the_boundary

			if len(new_indices_external_faces_thickness_side) != self.num_external_faces_for_boundary_condition:
				raise ValueError(" ❌ ERROR: There is part of the boundary that is missing the specification of the boundary condition. len(new_indices_external_faces_thickness_side) (== %d) != self.num_external_faces_for_boundary_condition (== %d)." %(len(new_indices_external_faces_thickness_side), self.num_external_faces_for_boundary_condition))

			# Reorder the necessary variables for the external faces that changed

			external_faces = self._external_faces.copy()
			external_face_neighbours = self._external_face_neighbours.copy()

			for i in range(len(new_indices_external_faces_thickness_side)):

				current_index = self.num_external_faces_frontside_backside + i
				index_to_swap = new_indices_external_faces_thickness_side[i]

				external_faces[current_index] = self._external_faces[index_to_swap]
				external_face_neighbours[current_index] = self._external_face_neighbours[index_to_swap]

			self._external_faces = external_faces
			self._external_face_neighbours = external_face_neighbours

		elif self.domain_type == '3D':

			############### Set the boundary conditions ############

			new_indices_external_faces = []
			for mesh_function_tag in mesh_function_tag_to_boundary_name:

				# startFace of the boundary
				current_boundary_position = len(new_indices_external_faces)

				# Name of the boundary
				boundary_name = mesh_function_tag_to_boundary_name[mesh_function_tag]
				utils.customPrint(" ╎    - Boundary: '%s'..." %(boundary_name))

				if boundary_name == 'unset': # It's a dummy marker. Please, don't leave anything marked in it!
					pass
				else:

					# Find the indices of the boundary
					mesh_indices_faces_of_the_boundary = mesh_function.where_equal(mesh_function_tag)
					indices_faces_of_the_boundary = [self._map_from_mesh_face_index_to_external_face_index[i_face] for i_face in mesh_indices_faces_of_the_boundary]

					# Include the boundary information specified by the user
					self._boundary_information[boundary_name] = {}
					assert 'type' in boundary_data['boundaries'][boundary_name]

					for key in boundary_data['boundaries'][boundary_name]:
						self._boundary_information[boundary_name][key] = boundary_data['boundaries'][boundary_name][key]

					# Include the computed boundary location
					self._boundary_information[boundary_name]['nFaces'] = len(indices_faces_of_the_boundary)
					self._boundary_information[boundary_name]['startFace'] = self.num_internal_faces + current_boundary_position

					# Update
					new_indices_external_faces += indices_faces_of_the_boundary

			if len(new_indices_external_faces) != len(self._external_faces):
				raise ValueError(" ❌ ERROR: There is part of the boundary that is missing the specification of the boundary condition. len(new_indices_external_faces) (== %d) != len(self._external_faces) (== %d)." %(len(new_indices_external_faces), len(self._external_faces)))

			# Reorder the necessary variables for the external faces that changed

			external_faces = self._external_faces.copy()
			external_face_neighbours = self._external_face_neighbours.copy()

			for i in range(len(new_indices_external_faces)):

				current_index = i
				index_to_swap = new_indices_external_faces[i]

				external_faces[current_index] = self._external_faces[index_to_swap]
				external_face_neighbours[current_index] = self._external_face_neighbours[index_to_swap]

			self._external_faces = external_faces
			self._external_face_neighbours = external_face_neighbours

		########### Periodic ("cyclic") boundary condition #############
		# Only one periodic boundary SubDomain is supported, because
		 # FEniCS only supports one (* In the creation of a FunctionSpace ("constrained_domain")).
		 # * Anyway, if you are using "cyclicAMI", the code will not do anything here.
		periodic_boundary_subdomain = boundary_data.get('periodic_boundary_subdomain', None)

		if type(periodic_boundary_subdomain).__name__ != 'NoneType':

			cyclic_boundaries_already_considered = []
			for boundary_name in self._boundary_information:

				# Periodic ("cyclic") boundary condition
				if boundary_name in ['frontAndBackSurfaces', 'frontSurface', 'backSurface']:
					pass
				elif (boundary_data['boundaries'][boundary_name]['type'] == 'cyclic') and (boundary_name not in cyclic_boundaries_already_considered):

					utils.customPrint(" ╎    -> Adjusting face order for periodic (\"cyclic\") boundary condition...")
					 # * Reorder the necessary external faces in order to match the boundary condition

					# Periodic SubDomain
					def mapVertexToDestination(vertex_id):
						"""
						Maps a vertex to the destination.
						"""

						vertex_coordinates = self._coordinates[vertex_id]
						if self.domain_type == '2D':
							x = np.array([vertex_coordinates[0], vertex_coordinates[1]]) # (x, y)
							z_coord = vertex_coordinates[2] # z
							y = np.asarray(x) # (x, y)
							periodic_boundary_subdomain.map(x, y)
							y_mapped = np.array([y[0], y[1], z_coord]) # (x, y, z)

						elif self.domain_type == '2D axisymmetric':
							x = np.array([vertex_coordinates[0], vertex_coordinates[2]]) # (r, z)
							theta_coord = vertex_coordinates[1] # θ
							y = np.asarray(x) # (r, z)
							periodic_boundary_subdomain.map(x, y)
							y_mapped = np.array([y[0], theta_coord, y[1]]) # (r, θ, z)

						elif self.domain_type == '3D':
							x = vertex_coordinates # (x, y, z)
							y = np.asarray(x) # (x, y, z)
							periodic_boundary_subdomain.map(x, y)
							y_mapped = y # (x, y, z)

						# Destination coordinates
						destination_coordinates = y_mapped

						# Find the mapped vertex
						mapped_vertex = utils.findNearestSubArray(self._coordinates, destination_coordinates, parameters_to_return = 'index')

						return mapped_vertex

					def checkVertexOnDestination(vertex_id):
						"""
						Check if a vertex is on the destination.
						"""

						vertex_coordinates = self._coordinates[vertex_id]
						if self.domain_type == '2D':
							coords_to_check = np.array([vertex_coordinates[0], vertex_coordinates[1]]) # (x, y)

						elif self.domain_type == '2D axisymmetric':
							coords_to_check = np.array([vertex_coordinates[0], vertex_coordinates[2]]) # (r, z)

						elif self.domain_type == '3D':
							coords_to_check = vertex_coordinates # (x, y, z)

						return periodic_boundary_subdomain.inside(coords_to_check, True)

					# Boundary 1
					nFaces1 = self._boundary_information[boundary_name]['nFaces']
					startFace1 = self._boundary_information[boundary_name]['startFace']
					boundary1_external_faces = [ startFace1 + i - self.num_internal_faces for i in range(nFaces1) ]

					# Boundary 2
					neighbour_boundary_name = self._boundary_information[boundary_name]['neighbourPatch']
					nFaces2 = self._boundary_information[neighbour_boundary_name]['nFaces']
					startFace2 = self._boundary_information[neighbour_boundary_name]['startFace']
					boundary2_external_faces = [ startFace2 + i - self.num_internal_faces for i in range(nFaces2) ]

					#### Check which of the boundaries is the one that is "mapped to" by the 'periodic_boundary_subdomain'

					# Coordinates of one vertex of Boundary 1
					boundary1_one_vertex = self._external_faces[boundary1_external_faces[0]][0]

					if checkVertexOnDestination(boundary1_one_vertex) == True:
						startFace_destination = startFace1
						boundary_destination_external_faces = boundary1_external_faces
						boundary_origin_external_faces = boundary2_external_faces
					else:
						startFace_destination = startFace2
						boundary_destination_external_faces = boundary2_external_faces
						boundary_origin_external_faces = boundary1_external_faces

					# All vertices of the 'origin'
					all_origin_vertices = []
					for i_face in boundary_origin_external_faces:
						all_origin_vertices += self._external_faces[i_face].tolist()
					all_origin_vertices = utils.removeRepeatedElementsFromList(all_origin_vertices, keep_order = True)

					# Create a map of all vertices of the 'origin' to the 'destination'
					map_vertex_from_origin_to_destination = {origin_vertex : mapVertexToDestination(origin_vertex) for origin_vertex in all_origin_vertices}

					#### Use string identificators
					# * This is because now we only have IDs, which are int numbers and not float.

					def createFaceStringID(face_vertices):

						# Sort the face vertices in ascending order
						face_vertices.sort(reverse = False) 

						return str(face_vertices)

					# Map the origin face vertices to destination
					mapped_origin_external_faces_vertices_str = [createFaceStringID([map_vertex_from_origin_to_destination[vertex_id] for vertex_id in self._external_faces[origin_face]]) for origin_face in boundary_origin_external_faces]

					# Destination
					destination_external_faces_vertices_str = [createFaceStringID(self._external_faces[destination_face].tolist()) for destination_face in boundary_destination_external_faces]

					# For all faces, find the face that matches all mapped vertices
					new_order_for_the_destination_external_faces = []

					utils.printProgress(0, max_iterations = len(mapped_origin_external_faces_vertices_str))

					for i_origin_rel in range(len(mapped_origin_external_faces_vertices_str)):

						origin_face_str = mapped_origin_external_faces_vertices_str[i_origin_rel]
						destination_face_indices_str = utils.searchMatchingElements([origin_face_str], destination_external_faces_vertices_str, profile_key = 'periodic boundary condition search')

						if len(destination_face_indices_str) == 1:
							pass # OK, it is fine!
						elif len(destination_face_indices_str) == 0:
							utils.customPrint(" ❌ ERROR: Face not mapped from origin! origin_face_str = %s, destination_face_indices_str = %s" %(origin_face_str, destination_face_indices_str))

							indices = np.fromstring(origin_face_str[i][1:len(origin_face_str[i]) - 1], dtype = 'int', sep = ',')
							for k in range(len(indices)):
								utils.customPrint(" -> self._coordinates[", origin_face_str[k], "] => ", self._coordinates[origin_face_str[k]])

							raise ValueError(" ❌ ERROR: Non-matching face found! Check above for more information!")

						else:
							raise ValueError(" ❌ ERROR: Too many mapped faces (%d)! destination_face_indices_str = %s" %(len(destination_face_indices_str), destination_face_indices_str))

						i_destination = destination_external_faces_vertices_str.index(destination_face_indices_str[0])

						new_order_for_the_destination_external_faces += [i_destination]

						utils.printProgress(i_origin_rel + 1)

					#### Apply the map to reorder the external faces in order for it to match the periodic boundary

					external_faces = self._external_faces.copy()
					external_face_neighbours = self._external_face_neighbours.copy()

					for i in range(len(new_order_for_the_destination_external_faces)):

						current_index = i + startFace_destination - self.num_internal_faces
						index_to_swap = new_order_for_the_destination_external_faces[i] + startFace_destination - self.num_internal_faces
						if current_index != index_to_swap:
							external_faces[current_index] = self._external_faces[index_to_swap]
							external_face_neighbours[current_index] = self._external_face_neighbours[index_to_swap]

					self._external_faces = external_faces
					self._external_face_neighbours = external_face_neighbours

					# Add to cyclic boundaries that are already considered
					cyclic_boundaries_already_considered += [ boundary_name, neighbour_boundary_name ]

	################################# Data #################################

	def boundary_information(self):
		if '_boundary_information' not in self.__dict__:
			raise ValueError(" ❌ ERROR: No boundary information computed yet!!")
		else:
			return self._boundary_information.copy()

	def coordinates(self):
		return self._coordinates.copy()

	def internal_faces(self):
		return self._internal_faces.copy()

	def external_faces(self):
		return self._external_faces.copy()

	def internal_face_neighbours(self):
		return self._internal_face_neighbours.copy()

	def external_face_neighbours(self):
		return self._external_face_neighbours.copy()

########################## MeshCreatorFromFEniCStoFoam #########################

class MeshCreatorFromFEniCStoFoam(FoamMeshWriter):
	"""
	Create a mesh in OpenFOAM from a FEniCS mesh.
	"""

	def __init__(self, mesh, boundary_data, problem_folder, domain_type = '2D', ignore_boundary_information = False):

		self.mesh = mesh
		self.boundary_data = boundary_data
		self.problem_folder = problem_folder
		self.domain_type = domain_type

		################# Initial verification #########################

		if (mesh.topology().dim() == 2 and (self.domain_type == '2D'  or self.domain_type == '2D axisymmetric')):
			pass
		elif mesh.topology().dim() == 3 and self.domain_type == '3D':
			pass
		else:
			raise ValueError(" ❌ ERROR: mesh.topology().dim() == %d is not defined!" %(mesh.topology().dim()))

		####################### Mesh3DtoFoam ###########################

		utils.printDebug(" ╎ 🌀 Creating auxiliary mesh structure for converting to OpenFOAM mesh...")

		mesh_3D_to_foam = Mesh3DtoFoam(mesh, domain_type = domain_type)

		if type(boundary_data).__name__ != 'NoneType' and ignore_boundary_information == False:

			if utils_fenics_mpi.runningInParallel():

				# Local boundary data
				local_boundary_data = boundary_data

				# Prepare another
				unified_boundary_data = {key : local_boundary_data[key] for key in local_boundary_data if key != 'mesh_function'}

				# Things to prepare beforehand, since they won't be accessible
				 # when computing the rest in a single processor.
				prepare_beforehand_mesh = [
					'array',
				]

				# Unify the mesh_function
				unified_mesh_function = utils_fenics_mpi.UnifiedMeshFunction(boundary_data['mesh_function'], 
					mesh_3D_to_foam.unified_mesh,
					prepare_beforehand = prepare_beforehand_mesh,
					proc_destination = 'all'
					)
				unified_boundary_data['mesh_function'] = unified_mesh_function

				# Use the unified_boundary_data
				boundary_data = unified_boundary_data

		# [Parallel]
		if utils_fenics_mpi.runningInParallel():
			if utils_fenics_mpi.runningInFirstProcessor():
				with utils_fenics_mpi.first_processor_lock():
					pass # Continuing with only the first processor
			else:
				return # OK. Let's wrap up and then wait

		if type(boundary_data).__name__ != 'NoneType' and ignore_boundary_information == False:
			mesh_3D_to_foam.setBoundaryInformation(boundary_data, boundaries_to_consider = 'external')

		########################## points ##############################
		# points: List of vectors describing the cell vertices, where the first vector in the list represents vertex 0, the second vector represents vertex 1, etc. 
			 # https://cfd.direct/openfoam/user-guide/v7-mesh-description/

		utils.printDebug(" ╎ ")
		utils.printDebug(" ╎ 🌀 Setting OpenFOAM mesh: 'points'")
		points_array = mesh_3D_to_foam.coordinates()

		########################### faces ##############################
		# faces: List of faces, each face being a list of indices to vertices in the points list, where again, the first entry in the list represents face 0, etc.
			# -> Ordering: First: internal faces; Later: external faces
			 # https://cfd.direct/openfoam/user-guide/v7-mesh-description/

		utils.printDebug(" ╎ 🌀 Setting OpenFOAM mesh: 'faces'")
		external_faces = mesh_3D_to_foam.external_faces()
		internal_faces = mesh_3D_to_foam.internal_faces()
		faces_array = utils.numpy_append_with_multiple_dimensions(internal_faces, external_faces, axis = 0)

		########################### owner ##############################
		# owner: List of owner cell labels, the index of entry relating directly to the index of the face, so that the first entry in the list is the owner label for face 0, the second entry is the owner label for face 1, etc
			# -> The owner cell should be the lowest numbered cell (http://openfoamwiki.net/index.php/Write_OpenFOAM_meshes)
			# -> Ordering: First: internal faces; Later: external faces
			 # https://cfd.direct/openfoam/user-guide/v7-mesh-description/

		utils.printDebug(" ╎ 🌀 Setting OpenFOAM mesh: 'owner'")
		internal_face_neighbours = mesh_3D_to_foam.internal_face_neighbours()
		external_face_neighbours = mesh_3D_to_foam.external_face_neighbours()

		lowest_numbered_internal_face_neighbours = internal_face_neighbours[:,0]

		owner_array = utils.numpy_append_with_multiple_dimensions(lowest_numbered_internal_face_neighbours, external_face_neighbours)
			# internal_face_neighbours[:,0]: The first cell of the face corresponds to the lowest numbered cell of the face
			# external_face_neighbours[:,0]: There is only one cell for each face anyway

		######################### neighbour ############################
		# neighbour: List of neighbour cell labels.
			# -> Only internal faces (http://openfoamwiki.net/index.php/Write_OpenFOAM_meshes).
			# -> The cells are created by defining an owner cell for each of the faces and also a neighbour cell for faces between two cells. (http://serpent.vtt.fi/mediawiki/index.php/Unstructured_mesh_based_input)
			 # https://cfd.direct/openfoam/user-guide/v7-mesh-description/

		utils.printDebug(" ╎ 🌀 Setting OpenFOAM mesh: 'neighbour'")
		highest_numbered_internal_face_neighbours = internal_face_neighbours[:,1]
		neighbour_array = highest_numbered_internal_face_neighbours

		######################### boundary #############################
		# boundary: List of patches, containing a dictionary entry for each patch, declared using the patch name, e.g. 
			#movingWall
			#{
			#    type patch;
			#    nFaces 20;
			#    startFace 760;
			#}
			#The startFace is the index into the face list of the first face in the patch, and nFaces is the number of faces in the patch.
			 # https://cfd.direct/openfoam/user-guide/v7-mesh-description/

		utils.printDebug(" ╎ 🌀 Setting OpenFOAM mesh: 'boundary'")

		if type(boundary_data).__name__ != 'NoneType' and ignore_boundary_information == False:
			boundary_dictionary = mesh_3D_to_foam.boundary_information()
		else:
			boundary_dictionary = {}

		self.points_array = points_array
		self.faces_array = faces_array
		self.neighbour_array = neighbour_array
		self.owner_array = owner_array
		self.boundary_dictionary = boundary_dictionary

		self.num_cells = mesh_3D_to_foam.num_cells
		self.num_faces = mesh_3D_to_foam.num_faces
		self.num_internal_faces = mesh_3D_to_foam.num_internal_faces
		self.num_points = mesh_3D_to_foam.num_vertices


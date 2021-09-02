################################################################################
#                               FoamMeshWriter                                 #
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
from ..io.FoamWriter import printContentSeparatedBySingleSpaces

# Utilities
from ..utils import utils

# OpenFOAM information
from ..utils import foam_information

############################### write_start_end ################################

def write_start_end(name):
	"""
	Decorator to write start, end and to file.

	-> Example of usage:
		@write_start_end('points')
		def f(*args, **kwargs):
			[...]
	"""

	def decorator(original_function):
		def wrapper(*args, **kwargs):

			# The first argument is 'self'
			self = args[0]

			# Include the file header
			# Header
			text = """%s""" %(foam_information._file_header_)

			# Content
			text += original_function(*args, **kwargs)

			# Footer
			text += """

// ************************************************************************* //"""

			utils.createFolderIfItDoesntExist("%s/constant/polyMesh" %(self.problem_folder), mpi_wait_for_everyone = False)
			file_path = "%s/constant/polyMesh/%s" % (self.problem_folder, name)
			utils.printDebug(" ╎      📝 Writing %s to %s" %(name, file_path))
			with open(file_path, "w", encoding = 'utf-8') as text_file:
				text_file.write(text)

			return 

		return wrapper

	return decorator

################################## FoamMeshWriter ##############################

class FoamMeshWriter():
	"""
	Write a mesh that is already in the OpenFOAM format to OpenFOAM files.
	"""

	def __init__(self, foam_mesh):

		utils.printDebug("\n 🌀 Creating the OpenFOAM mesh from a FoamMesh...")

		self.points_array = foam_mesh.data['points']['points']
		self.faces_array = foam_mesh.data['faces']['faces']
		self.neighbour_array = foam_mesh.data['neighbour']['neighbour']
		self.owner_array = foam_mesh.data['owner']['owner']
		self.boundary_dictionary = foam_mesh.data['boundary']['boundary']

		self.num_cells = foam_mesh.num_cells()
		self.num_faces = foam_mesh.num_faces()
		self.num_internal_faces = foam_mesh.num_internal_faces()
		self.num_points = foam_mesh.num_vertices()

	############################### write_to_foam ##########################

	@utils.only_the_first_processor_executes(wait_for_everyone = False)
	def write_to_foam(self, python_write_precision = 6):
		"""
		Write OpenFOAM mesh to files.
		"""

		assert 'int' in type(python_write_precision).__name__

		utils.customPrint(" ╎ ")
		utils.customPrint(" ╎ 🌀 Writing OpenFOAM mesh to files...")

		utils.customPrint(" ╎   - points")
		self.write_points(python_write_precision = python_write_precision)

		utils.customPrint(" ╎   - owner")
		self.write_owner(python_write_precision = python_write_precision)

		utils.customPrint(" ╎   - neighbour")
		self.write_neighbour(python_write_precision = python_write_precision)

		utils.customPrint(" ╎   - faces")
		self.write_faces(python_write_precision = python_write_precision)

		utils.customPrint(" ╎   - boundary")
		self.write_boundary(python_write_precision = python_write_precision)

	@write_start_end('points')
	def write_points(self, python_write_precision = 6):
		"""
		Write points.
		"""

		# FoamFile definition
		text = """FoamFile
{
    version     2.0;
    format      ascii;
    class       vectorField;
    location    "constant/polyMesh";
    object      points;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""

		# Print the array of points
		text += """%d
(
""" %(len(self.points_array))
		for point in self.points_array:

			val_text = ""
			for i in range(len(point)):
				if i != 0:
					val_text += " "
				val_text += "%1.*e" %(python_write_precision, point[i])

			text += "(%s)\n" %(val_text)

		text += ")"

		return text

	@write_start_end('owner')
	def write_owner(self, python_write_precision = 6):
		"""
		Write owner.
		"""

		# FoamFile definition
		text = """FoamFile
{
    version     2.0;
    format      ascii;
    class       labelList;
    note        "nPoints: %d nCells: %d nFaces: %d nInternalFaces: %d";
    location    "constant/polyMesh";
    object      owner;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

""" %(self.num_points, self.num_cells, self.num_faces, self.num_internal_faces)

		# Print the array of owner
		text += """%d
(
""" %(len(self.owner_array))
		for point in self.owner_array:
			text += "%d\n" %(point)

		text += ")"

		return text

	@write_start_end('neighbour')
	def write_neighbour(self, python_write_precision = 6):
		"""	
		Write neighbour.
		"""

		# FoamFile definition
		text = """FoamFile
{
    version     2.0;
    format      ascii;
    class       labelList;
    note        "nPoints: %d nCells: %d nFaces: %d nInternalFaces: %d";
    location    "constant/polyMesh";
    object      neighbour;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

""" %(self.num_points, self.num_cells, self.num_faces, self.num_internal_faces)

		# Print the array of neighbours
		text += """%d
(
""" %(len(self.neighbour_array))
		for point in self.neighbour_array:
			text += "%d\n" %(point)

		text += ")"

		return text

	@write_start_end('faces')
	def write_faces(self, python_write_precision = 6):
		"""
		Write faces.
		"""

		# FoamFile definition
		text = """FoamFile
{
    version     2.0;
    format      ascii;
    class       faceList;
    location    "constant/polyMesh";
    object      faces;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""

		# Print the array of faces
		text += """%d
(
""" %(len(self.faces_array))
		for point in self.faces_array:

			val_text = ""
			for i in range(len(point)):
				if i != 0:
					val_text += " "
				val_text += "%d" %(point[i])

			text += "%s(%s)\n" %(len(point), val_text)

		text += ")"

		return text

	@write_start_end('boundary')
	def write_boundary(self, python_write_precision = 6):
		"""
		Write boundary.
		"""

		# FoamFile definition
		text = """FoamFile
{
    version     2.0;
    format      ascii;
    class       polyBoundaryMesh;
    location    "constant/polyMesh";
    object      boundary;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""

		# Print the dictionary of boundaries
		text += """%d
(
""" %(len(self.boundary_dictionary))
		for key in self.boundary_dictionary:
			text += "    %s\n" %(key)
			text += "    {\n"
			for key2 in self.boundary_dictionary[key]:
				if type(self.boundary_dictionary[key][key2]).__name__ == 'list':
					text += "        %-15s List<word> %d(" %(key2, len(self.boundary_dictionary[key][key2]))

					# Write the content of the list separated by single spaces
					text += printContentSeparatedBySingleSpaces(self.boundary_dictionary[key][key2])

					text += ");\n"

					#text += "        %-15s %s;\n" %(key2, self.boundary_dictionary[key][key2])
				else:
					text += "        %-15s %s;\n" %(key2, self.boundary_dictionary[key][key2])
			text += "    }\n"
		text += ")"

		return text


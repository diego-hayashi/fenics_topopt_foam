################################################################################
#                                 FoamWriter                                   #
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

############################# Project libraries ################################

# Utilities
from ..utils import utils

# OpenFOAM information
from ..utils import foam_information

################################## FoamWriter ##################################

class FoamWriter():
	"""
	Writer for writing setup from Python dictionaries to OpenFOAM files.
	"""

	def __init__(self, problem_folder, python_write_precision = 6):

		# Problem folder
		self.problem_folder = problem_folder

		# Write precision
		assert 'int' in type(python_write_precision).__name__
		self.python_write_precision = python_write_precision

		# Folders
		self.properties_folder = "%s/constant" %(self.problem_folder)
		self.configuration_folder = "%s/system" %(self.problem_folder)

	####################### create_folder_structure ########################

	def create_folder_structure(self):
		"""
		Create folder with the OpenFOAM file structure and the complete definition of the problem:

			 https://cfd.direct/openfoam/user-guide/v7-case-file-structure/#x16-1220004.1
			 📂 [case]
			 ├ 📂 system     -> Solver parameters
			 │  ├ 📃 controlDict ->  "Control dictionary"    -> Run control parameters are set including start/end time, time step and parameters for data output
			 │  ├ 📃 fvSchemes   -> "Finite Volume Schemes"  -> Discretisation schemes
			 │  ╰ 📃 fvSolution  -> "Finite Volume Solution" -> equation solvers, tolerances and other algorithm controls
			 │ 
			 ├ 📂 constant
			 │  ├ 📃 [xxx]Properties -> Properties of the "xxx" model of the material. There may be lots of files as this one.
			 │  ╰ 📂 polyMesh -> Mesh of the problem
			 │     ├ 📃 points 
			 │     ├ 📃 faces
			 │     ├ 📃 owner
			 │     ├ 📃 neighbour
			 │     ╰ 📃 boundary
			 │ 
			 ╰ 📂 [time directories] -> Called: 0, 1, 2 ... (or with the time itself as 0.000000e+00, 0.000120e+00 ...)

		"""

		utils.createFolderIfItDoesntExist("%s" %(self.problem_folder))
		utils.createFolderIfItDoesntExist("%s/0" %(self.problem_folder))
		utils.createFolderIfItDoesntExist("%s/constant" %(self.problem_folder))
		utils.createFolderIfItDoesntExist("%s/system" %(self.problem_folder))

	############################### FoamVector #############################

	def writeFoamVector(self, foam_vector, time_step_name = '0'):
		"""
		Write FoamVector to file.
		"""

		data_to_add = foam_vector.data
		foam_vector_name = foam_vector.name
		assert foam_vector_name != ""

		if foam_vector.type == 'volScalarField':
			prefer_scalar_nonuniform_lists = True
		elif foam_vector.type == 'volVectorField':
			prefer_scalar_nonuniform_lists = False
		else:
			raise ValueError(" %s ❌ ERROR: FoamVector of type '%s' is not defined!" %(foam_vector.type))

		self.writeDataToFile(
			data_to_add = data_to_add, 
			file_to_use = "%s/%s/%s" % (self.problem_folder, time_step_name, foam_vector_name),
			prefer_scalar_nonuniform_lists = prefer_scalar_nonuniform_lists
		)

	############################## FoamProperty ############################

	def writeFoamProperty(self, foam_property):
		"""
		Write FoamProperty to file.
		"""

		data_to_add = foam_property.data
		foam_property_name = foam_property.name
		assert foam_property_name != ""

		self.writeDataToFile(
			data_to_add = data_to_add, 
			file_to_use = "%s/%s" % (self.properties_folder, foam_property_name)
		)

	############################ FoamConfiguration #########################

	def writeFoamConfiguration(self, foam_configuration):
		"""
		Write FoamConfiguration to file.
		"""

		data_to_add = foam_configuration.data
		foam_configuration_name = foam_configuration.name
		assert foam_configuration_name != ""

		self.writeDataToFile(
			data_to_add = data_to_add, 
			file_to_use = "%s/%s" % (self.configuration_folder, foam_configuration_name)
		)

	############################# writeDataToFile ##########################

	@utils.only_the_first_processor_executes()
	def writeDataToFile(self, data_to_add = None, preferred_parameter_order = [], file_to_use = None, prefer_scalar_nonuniform_lists = False):
		"""
		Write data to file.
		"""

		# File name
		file_to_use_split = utils.removeTrailingSlash(file_to_use).split('/')
		file_name = file_to_use_split[len(file_to_use_split) - 1]
		folder_name = file_to_use_split[len(file_to_use_split) - 2]

		# Preferred parameter order
		if folder_name != 'constant' and folder_name != 'system':
			plus_preferred_parameter_order = ["FoamFile", "dimensions", "internalField", "boundaryField"]
		elif file_name == 'transportProperties':
			plus_preferred_parameter_order = ["FoamFile", "transportModel", "nu"]
		elif file_name == 'turbulenceProperties':
			plus_preferred_parameter_order = ["FoamFile", "simulationType", "RAS"]
		else:
			plus_preferred_parameter_order = ["FoamFile"]
		preferred_parameter_order = plus_preferred_parameter_order + preferred_parameter_order

		# Sections
		sections = utils.order_list_from_preferred_order(
			list(data_to_add.keys()), 
			preferred_order = preferred_parameter_order
			)
		assert "FoamFile" in sections
		assert preferred_parameter_order[0] == "FoamFile"

		# Include the file header
		contents = "%s" %(foam_information._file_header_)

		first_data_added_flag = False
		for section_name in sections:
			contents += self.print_section_to_text(data_to_add[section_name], section_name, prefer_scalar_nonuniform_lists = prefer_scalar_nonuniform_lists)
			if first_data_added_flag == False: # * The first section added is the FoamFile
				first_data_added_flag = True
				contents += """// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"""

		contents += """// ************************************************************************* //"""

		with open("%s" %(file_to_use), "w", encoding = 'utf-8') as text_file:
			text_file.write(contents)

	def print_section_to_text(self, data, section_name, indentation_level = 0, prefer_scalar_nonuniform_lists = False):
		"""
		Print section to text.
		"""

		if indentation_level == 0: 
			column_size = 15
		else:
			column_size = 11

		# If it is a 'SuccessiveSimulationParameter', extract the current value.
		if type(data).__name__ == 'SuccessiveSimulationParameter':
			data = data.value()

		text = ""
		if type(data).__name__ == 'dict':

			# Section title
			if section_name != '':
				text += "%s%s\n" %("    "*indentation_level, section_name)

			# Start section
			text += "%s{\n" %("    "*indentation_level)

			# For each key in the dictionary
			if section_name == 'FoamFile':
				preferred_order = ["version", "format", "class", "location", "object"]
			elif section_name == 'RAS':
				preferred_order = ["RASModel", "turbulence", "printCoeffs"]
			else:
				preferred_order = []

			ordered_keys = utils.order_list_from_preferred_order(
				list(data.keys()), 
				preferred_order = preferred_order
			)
			for key in ordered_keys:
				text += self.print_section_to_text(data[key], key, indentation_level = indentation_level + 1, prefer_scalar_nonuniform_lists = prefer_scalar_nonuniform_lists) # Recursion
			#for key in data:
			#	text += self.print_section_to_text(data[key], key, indentation_level = indentation_level + 1, prefer_scalar_nonuniform_lists = prefer_scalar_nonuniform_lists) # Recursion

			# Finish section
			if section_name == 'FoamFile':
				text += "%s}\n" %("    "*indentation_level)
			else:
				text += "%s}\n\n" %("    "*indentation_level)

		elif type(data).__name__ in ['bool']:
			# https://www.openfoam.com/documentation/guides/latest/doc/openfoam-guide-input-types.html

			if data == True:
				data_to_write = 'true' # Or 'on'
			else:
				data_to_write = 'false' # Or 'off'

			# Write as string
			text += "%s%-*s %s;\n" %("    "*indentation_level, column_size, section_name, data_to_write)

			if indentation_level == 0: 
				text += '\n'

		elif type(data).__name__ in ['str', 'float', 'int']:

			# Write as string
			text += "%s%-*s %s;\n" %("    "*indentation_level, column_size, section_name, data)

			if indentation_level == 0: 
				text += '\n'

		elif type(data).__name__ == 'list':
			# Example: [[problem_folder]/0/p] 
				# dimensions      [0 2 -2 0 0 0 0];
				 # Corresponding list: [0, 2, -2, 0, 0, 0, 0]

			if all(isinstance(item, str) for item in data):
				text += "%s%-*s List<word> %d(" %("    "*indentation_level, column_size, section_name, len(data))

				# Write the content of the list separated by single spaces
				text += printContentSeparatedBySingleSpaces(data, python_write_precision = self.python_write_precision)

				text += ");\n"

			elif all(isinstance(item, dict) for item in data):
				#text += "%s%-*s (\n" %("    "*indentation_level, column_size, section_name)
				text += "%s%s\n\n(\n" %("    "*indentation_level, section_name)

				for i in range(len(data)):
					text += self.print_section_to_text(data[i], '', indentation_level = indentation_level + 1, prefer_scalar_nonuniform_lists = prefer_scalar_nonuniform_lists) # Recursion

				text += ");\n"

			else:
				text += "%s%-*s [" %("    "*indentation_level, column_size, section_name)

				# Write the content of the list separated by single spaces
				text += printContentSeparatedBySingleSpaces(data, python_write_precision = self.python_write_precision)

				text += "];\n"

			if indentation_level == 0: 
				text += '\n'

		elif type(data).__name__ == 'tuple':
			# Example: [[problem_folder]/constant/transportProperties] 
				# nu              [0 2 -1 0 0 0 0] 1e-05;
				 # Corresponding tuple: ([0, 2, -1, 0, 0, 0, 0], 1e-05)

			text += "%s%-*s" %("    "*indentation_level, column_size, section_name)

			# Write the content of the tuple separated by single spaces
			for i in range(len(data)):
				if type(data[i]).__name__ == 'list':

					# Write the content of the list separated by single spaces
					text += " ["
					text += printContentSeparatedBySingleSpaces(data[i], python_write_precision = self.python_write_precision)
					text += "]"

				else:
					text += " %s" %(data[i])

			text += ";\n"

			if indentation_level == 0: 
				text += '\n'

		elif type(data).__name__ == 'ndarray':

			if 'str' in data.dtype.name:

				text += "%s%-*s (" %("    "*indentation_level, column_size, section_name)
				text += printContentSeparatedBySingleSpaces(data, python_write_precision = self.python_write_precision)
				text += ");\n"

			else:

				# uniform data (volScalarField)
				if len(data.shape) == 1 and len(data) == 1:

					text += "%s%-*s uniform " %("    "*indentation_level, column_size, section_name)
					text += printContentSeparatedBySingleSpaces(data, python_write_precision = self.python_write_precision)
					text += ";\n"

				# uniform data (volVectorField)
				elif (len(data.shape) == 1 and len(data) == 3) and (prefer_scalar_nonuniform_lists == False):

					text += "%s%-*s uniform (" %("    "*indentation_level, column_size, section_name)
					text += printContentSeparatedBySingleSpaces(data, python_write_precision = self.python_write_precision)
					text += ");\n"

				# nonuniform data
				else:

					if len(data.shape) == 1: # nonuniform data (volScalarField)
						text += "%s%-*s nonuniform List<scalar>\n" %("    "*indentation_level, column_size, section_name)
					else: # nonuniform data (volVectorField)
						text += "%s%-*s nonuniform List<vector>\n" %("    "*indentation_level, column_size, section_name)
					text += "%d\n" %(len(data))
					text += "(\n"

					if len(data.shape) == 1:
						for i in range(len(data)):
							text += "%s\n" %(data[i])
					else:
						for i in range(len(data)):
							text += "("
							text += printContentSeparatedBySingleSpaces(data[i], python_write_precision = self.python_write_precision)
							text += ")\n"

					text += ")\n"
					text += ";\n"

			if indentation_level == 0: 
				text += '\n'

		else:
			raise ValueError(" ❌ ERROR: Type '%s' for {data == %s} is not recognized!" %(type(data).__name__, data))

		return text

def printContentSeparatedBySingleSpaces(data_list, python_write_precision = 6):
	"""
	Print content separated by single spaces.
	"""
	text = ""

	# Write the content of the list separated by single spaces
	for i in range(len(data_list)):
		if i >= 1:
			text += " "

		data_list_i = data_list[i]

		# If it is a 'SuccessiveSimulationParameter', extract the current value.
		if type(data_list_i).__name__ == 'SuccessiveSimulationParameter':
			data_list_i = data_list_i.value()

		if 'float' in type(data_list_i).__name__:
			text += "%1.*e" %(python_write_precision, data_list_i)
		else:
			text += "%s" %(data_list_i)

	return text





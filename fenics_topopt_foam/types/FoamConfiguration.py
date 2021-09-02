################################################################################
#                               FoamConfiguration                              #
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

############################## FoamConfiguration ###############################

class FoamConfiguration():
	"""
	Configurations for OpenFOAM.
	It is stored in [problem_folder]/system/[name].
	-> Some examples: 'fvSolution', 'fvSchemes', 'controlDict', 'fvOptions'
	"""

	def __init__(self, data, name = 'from data'):

		# Data (from FoamReader)
		self.reloadData(data, name)

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

	def reloadData(self, data, name, set_foam_file_info = False):
		"""
		Set the data obtained from the FoamReader.
		"""

		if name == 'from data':
			name = data['FoamFile']['object']

		if type(data).__name__ == 'NoneType' or len(data) == 0:

			if type(data).__name__ == 'NoneType':
				data = {}

			data['FoamFile'] = {
				'version' : '2.0',
				'format' : 'ascii',
				'class' : 'dictionary',
				'location' : '"system"',
				'object' : '%s' %(name),
			}

		else:
			if set_foam_file_info == True:

				data['FoamFile'] = {
					'version' : '2.0',
					'format' : 'ascii',
					'class' : 'dictionary',
					'location' : '"system"',
					'object' : '%s' %(name),
				}

			else:
				assert 'FoamFile' in data

		self.data = data

		# Flag that indicates if it is to apply changes
		self._APPLY_CHANGES = False

	################################ setFunction ###########################

	def setFunctionToConfiguration(self, function_name, function_data):
		"""
		Set function to the configuration.
		"""

		if 'functions' not in self.data:
			self.data['functions'] = {}

		self.data['functions'][function_name] = function_data

	############################# setConfiguration #########################

	def setConfiguration(self, configuration_dictionary):
		"""
		Set configurations.
		"""

		# Take off FoamFile from configuration_dictionary (if there is anything like that)
		new_data = configuration_dictionary.pop('FoamFile', {})

		# Data (from FoamReader)
		self.data.update(new_data)

	################################# name #################################

	@property
	def name(self):
		"""
		Name of the FoamConfiguration.
		"""
		return self.data['FoamFile']['object']




################################################################################
#                                FoamProperty                                  #
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

################################ FoamProperty ##################################

class FoamProperty():
	"""
	Property for OpenFOAM.
	It is stored in [problem_folder]/constant/[name].
	"""

	def __init__(self, data):

		# Data (from FoamReader)
		self.reloadData(data)

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

		# Flag that indicates if it is to apply changes
		self._APPLY_CHANGES = False

	################################ setProperty ###########################

	def setProperty(self, property_dictionary):
		"""
		Set properties.
		"""

		# Take off FoamFile from configuration_dictionary (if there is anything like that)
		new_data = property_dictionary.pop('FoamFile', {})

		# Data (from FoamReader)
		self.data.update(new_data)

	################################# name #################################

	@property
	def name(self):
		"""
		Name of the FoamProperty.
		"""
		return self.data['FoamFile']['object']


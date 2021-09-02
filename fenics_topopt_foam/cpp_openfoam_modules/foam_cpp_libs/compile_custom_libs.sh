#!/bin/bash

################################################################################
#                  🎦 Compile custom libraries for OpenFOAM 🎦                 #
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

# https://www.youtube.com/watch?v=MiUDCOhbQaM

# Please, set the OpenFOAM environment in bashrc before running this code!
# Run it as:
	# $ chmod +x compile_utilities_for_openfoam.sh
	# $ ./compile_utilities_for_openfoam.sh

echo

for lib_source_folder in "."/*; do

	# All folders here inside this folder MUST be library folders for compiling 
	lib_source_folder=${lib_source_folder:2} # Remove "./" from the beginning of the library folder
		# https://stackoverflow.com/questions/11469989/how-can-i-strip-first-x-characters-from-string-using-sed

	if [[ "$lib_source_folder" =~ "__" ]]; then
		: # Skip

	elif [ -d "$lib_source_folder" ]; then

		echo " 🎦 Library: $lib_source_folder"
		echo " ╎ 🌀 Cleaning up any previous compilation..."
		wclean $lib_source_folder

		echo " ╎ 🌀 Compiling library for OpenFOAM (generating .so file)..."
		wmake libso $lib_source_folder
		echo 
	fi
done

echo " ✅ All compiled!"
echo


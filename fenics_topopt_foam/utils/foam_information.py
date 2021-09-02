################################################################################
#                               foam_information                               #
################################################################################
# Some OpenFOAM information

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

import re

############################# Project libraries ################################

# Utilities
from ..utils import utils

############################## Foam information ################################

#print(" 🌀 Checking OpenFOAM information...")

# OpenFOAM version
try:

	# Try reading the OpenFOAM version

	#### [Old version]
	#_foam_version_ = utils.run_command_in_shell("echo -n $(foamVersion 2>&1)", mode = 'save output to variable', suppress_run_print = True) # Such as OpenFOAM-7, OpenFOAM-v1912 ...

	#### [New version] Reload the OpenFOAM environment in order to force foamVersion to be accessible
	_foam_version_ = utils.run_command_in_shell("source $FOAM_ETC/bashrc; echo -n $(foamVersion 2>&1)", mode = 'save output to variable', suppress_run_print = True) # Such as OpenFOAM-7, OpenFOAM-v1912 ...
	# * 2>&1 => Redirect stderr (2) to stdout (1). I don't know why foamVersion writes the version number to stderr instead of stdout

	if 'not found' in _foam_version_:
		raise ValueError(" ❌ ERROR: 'foamVersion' command failed! %s" %(_foam_version_))

except:

	# Print the traceback
	import traceback
	traceback.print_exc()

	# Warn the user that we are now using a 'default' name for the OpenFOAM version
	print(" ❗ Could not determine OpenFOAM version, because the command 'foamVersion' was not made accessible (as an 'export' function in shell). Assuming that you are running 'OpenFOAM-7'...")
	_foam_version_ = 'OpenFOAM-7' # Assuming 'OpenFOAM-7'

# OpenFOAM version number
__foam_version_split = _foam_version_.split('-')
_foam_version_number_ = __foam_version_split[len(__foam_version_split) - 1]

# OpenFOAM instalation directory
_foam_inst_dir_ = utils.run_command_in_shell("echo -n $FOAM_INST_DIR", mode = 'save output to variable', suppress_run_print = True)

# OpenFOAM source code directory
_foam_src_ = utils.run_command_in_shell("echo -n $FOAM_SRC", mode = 'save output to variable', suppress_run_print = True)

##################################### File header ##############################
 # It seems that OpenFOAM checks if we have the right header before running.
 # Therefore, we need to define it for each OpenFOAM implementation.

############################## OpenFOAM (openfoam.org) #########################
if re.match("(\d)", _foam_version_number_):
	_foam_implementation_ = 'OpenFOAM (openfoam.org)'
	_file_header_ = r"""/*--------------------------------*- C++ -*----------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  %s
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/
""" %(_foam_version_number_)

############################# OpenFOAM+ (openfoam.com) #########################
elif re.match("(v\d\d\d\d)", _foam_version_number_):
	_foam_implementation_ = 'OpenFOAM+ (openfoam.com)'
	_file_header_ = r"""/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  %s                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
""" %(_foam_version_number_)

########################## FOAM-extend (foam-extend.org) #######################
elif re.match("(\d.\d)", _foam_version_number_):
	_foam_implementation_ = 'FOAM-extend (foam-extend.org)'
	_file_header_ = r"""/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | foam-extend: Open Source CFD                    |
|  \\    /   O peration     | Version:     %s                                |
|   \\  /    A nd           | Web:         http://www.foam-extend.org         |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
""" %(_foam_version_number_)

else:
	raise ValueError(" ❌ ERROR: _foam_version_number_ == '%s' could not be mapped to an OpenFOAM version style!" %(_foam_version_number_))



#!/bin/bash

################################################################################
#                           🌀 Run the Python code 🌀                          #
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

# Create a log file
>log 

# Run with Python 3 (* Assuming that it is already the default in your machine),
 # append all prints to a log file, and return error if the code execution failed
(python -u topology_optimization_bend_pipe.py; ret=$?; exit $ret) 2>&1 | tee --append log


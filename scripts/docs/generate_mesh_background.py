#!/usr/bin/python3

################################################################################
#                       🛰️ Generate mesh background 🛰️                         #
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

import matplotlib.pyplot as plt

############################### FEniCS libraries ###############################

from fenics import *
from mshr import *

########################## Generate the mesh background ########################
# * Useful links:
 # https://stackoverflow.com/questions/9295026/matplotlib-plots-removing-axis-legends-and-white-spaces
 # https://stackoverflow.com/questions/46844963/python-matplotlib-print-to-resolution-and-without-white-space-borders-mar

# [mshr] Rectangle
rectangle = Rectangle(Point(0, 0), Point(1,2))

# [mshr] Generate the mesh
resolution = 8#16
mesh = generate_mesh(rectangle, resolution)

# Create the figure
fig = plt.figure(0)

# Get the axes
ax = plt.gca()

# Plot the mesh
plotted_object = plot(mesh, color = '#e4edf8')

# Set background color
ax.set_facecolor('#fdfdfd')

# Remove margins
plt.margins(0,0)

# Turn off axes
plt.axis("off")

# Remove the white border
plt.axis("tight")

# Square up the image instead of filling the "figure" space
plt.axis("image")

# Make the layout tight
plt.tight_layout() 

# Save the figure
plt.savefig('docs/img/mesh_background.png', bbox_inches = 'tight', pad_inches = -0.1, dpi = 300)

# Close the figure
plt.close()



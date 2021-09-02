/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    cellCenters2Dtriangles

Description
    Write the position vector of the 2D cell centers to the corresponding time directory. 

Modified by
    Diego Hayashi Alonso (2020-2021)

Based on
    https://bitbucket.org/peterjvonk/cellcenters

\*---------------------------------------------------------------------------*/

// Argument list
#include "argList.H"

// Time selector
#include "timeSelector.H"

// Time
#include "Time.H"

// Finite volumes
#include "fvMesh.H"

// Vector I/O field
#include "vectorIOField.H"

// Volumetric fields
#include "volFields.H"

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[]){

	// Add options
	timeSelector::addOptions();

	// Add region
	#include "addRegionOption.H"

	// Set case
	#include "setRootCase.H"

	// Create time
	#include "createTime.H"

	// List of times
	instantList timeDirs = timeSelector::select0(runTime, args);

	// Create the mesh
	#include "createNamedMesh.H"

	// For all specified time folders
	forAll(timeDirs, timeI){

		// Set the current time
		runTime.setTime(timeDirs[timeI], timeI);

		// Print the current time
		Info<< "Time = " << runTime.timeName() << endl;

		// Update mesh
		mesh.readUpdate();

		// Compute cellCenters2D

		volVectorField cellCenters2D(
			IOobject(
				"cellCenters2Dtriangles",
				runTime.timeName(),
				mesh,
				IOobject::NO_READ,
				IOobject::AUTO_WRITE
			),
			mesh,
			vector(0,0,0)
		);

		vector current_coords = vector(0,0,0);
		vector center2D_coords = vector(0,0,0);
		int count_here = 0;

		Info<< "Computing cellCenters2D in " << runTime.timeName() << endl;
		forAll(mesh.cells(),cellI){

			center2D_coords.x() = 0;
			center2D_coords.y() = 0;
			center2D_coords.z() = 0;
			count_here = 0;

			// Compute triangle centroid -- https://en.wikipedia.org/wiki/Centroid#Of_a_triangle
			const labelList& cellPts = mesh.cellPoints()[cellI];

			forAll(cellPts,cellPtI){
				current_coords = mesh.points()[cellPts[cellPtI]];
				if (current_coords.y() >= 0.0){
					center2D_coords.x() += current_coords.x();
					center2D_coords.z() += current_coords.z();
					count_here += 1;
				}
			}
			cellCenters2D[cellI] = center2D_coords/count_here;
		} 

		// Write cellCenters2D to file
		Info<< "Writing cellCenters2D to " << cellCenters2D.name() << " in " << runTime.timeName() << endl;
		cellCenters2D.write();

	}

	Info<< "\nEnd\n" << endl;

	return 0;
}


// ************************************************************************* //

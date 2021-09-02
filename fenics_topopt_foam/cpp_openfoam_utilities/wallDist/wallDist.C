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
    wallDist

Description
    Calculate and write the normal vector to walls and wall distance to files.

Modified by
    Diego Hayashi Alonso (2020-2021)

Based on
    OpenFOAM-7/applications/test/wallDist

\*---------------------------------------------------------------------------*/

// List of arguments
#include "argList.H"

// Time functions
#include "Time.H"

// Finite volume mesh
#include "fvMesh.H"

// Wall distance
#include "wallDist.H"

// Time selector
#include "timeSelector.H"

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[]){

	// Add options
	timeSelector::addOptions();

	// Set case
	#include "setRootCase.H"

	// Create time
	#include "createTime.H"

	// List of times
	instantList timeDirs = timeSelector::select0(runTime, args);

	// Create the mesh
	#include "createMesh.H"
	Info<< "Mesh read in = "<< runTime.cpuTimeIncrement()<< " s\n" << endl << endl;

	// For all specified time folders
	forAll(timeDirs, timeI){

		// Set the current time
		runTime.setTime(timeDirs[timeI], timeI);

		// Print the current time
		Info<< "Time = " << runTime.timeName() << endl;

		// Wall-reflection vectors -- Vectors normal to the walls
		Info<< "    Computing nWall" << endl;
		const volVectorField& n = wallDist::New(mesh).n();
		n.write();

		// Wall distance
		Info<< "    Computing yWall" << endl;
		const volScalarField& y = wallDist::New(mesh).y();
		y.write();

		// End
		Info<< endl;
	}

	return 0;
}


// ************************************************************************* //

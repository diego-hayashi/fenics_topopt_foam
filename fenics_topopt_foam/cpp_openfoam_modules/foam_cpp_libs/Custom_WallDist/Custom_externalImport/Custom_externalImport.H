/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2015-2019 OpenFOAM Foundation
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

Class
    Foam::patchDistMethods::Custom_externalImport

Description
    Wall distance loaded from file.

See also
    Foam::patchDistMethod::advectionDiffusion
    Foam::patchDistMethod::Poisson
    Foam::patchDistMethod::meshWave
    Foam::wallDist

SourceFiles
    Custom_externalImport.C

Modified by 
    Diego Hayashi Alonso (2020-2021)

Based on
    'OpenFOAM-7/src/finiteVolume/fvMesh/wallDist/patchDistMethods/advectionDiffusion/advectionDiffusionPatchDistMethod.H'

\*---------------------------------------------------------------------------*/

#ifndef Custom_externalImport_H
#define Custom_externalImport_H

#include "patchDistMethod.H"

// Add volVectorField to the environment
#include "volFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace patchDistMethods
{

/*---------------------------------------------------------------------------*\
                      Class Custom_externalImport Declaration
\*---------------------------------------------------------------------------*/

class Custom_externalImport
:
    public patchDistMethod
{
    // Private Member Data

        //- Sub-dictionary of coefficients
        dictionary coeffs_;

        // Wall distance from file
        volScalarField yWall_to_load_;

        // Normal vector from file
        volVectorField nWall_to_load_;

public:

    //- Runtime type information
    TypeName("Custom_externalImport");


    // Constructors

        //- Construct from coefficients dictionary, mesh
        //  and fixed-value patch set
        Custom_externalImport
        (
            const dictionary& dict,
            const fvMesh& mesh,
            const labelHashSet& patchIDs
        );

        //- Disallow default bitwise copy construction
        Custom_externalImport(const Custom_externalImport&) = delete;


    // Member Functions

        //- Correct the given distance-to-patch field
        virtual bool correct(volScalarField& y);

        //- Correct the given distance-to-patch and normal-to-patch fields
        virtual bool correct(volScalarField& y, volVectorField& n);


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const Custom_externalImport&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace patchDistMethods
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //

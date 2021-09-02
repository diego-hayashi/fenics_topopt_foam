/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2015-2018 OpenFOAM Foundation
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

Modified by 
    Diego Hayashi Alonso (2020-2021)

Based on
    'OpenFOAM-7/src/finiteVolume/fvMesh/wallDist/patchDistMethods/advectionDiffusion/advectionDiffusionPatchDistMethod.C'

\*---------------------------------------------------------------------------*/

#include "Custom_externalImport.H"
#include "surfaceInterpolate.H"
#include "fvcGrad.H"
#include "fvcDiv.H"
#include "fvmDiv.H"
#include "fvmLaplacian.H"
#include "fvmSup.H"
#include "addToRunTimeSelectionTable.H"

#include "fixedValueFvPatchFields.H"
#include "zeroGradientFvPatchFields.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace patchDistMethods
{
    defineTypeNameAndDebug(Custom_externalImport, 0);
    addToRunTimeSelectionTable(patchDistMethod, Custom_externalImport, dictionary);
}
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::patchDistMethods::Custom_externalImport::Custom_externalImport
(
    const dictionary& dict,
    const fvMesh& mesh,
    const labelHashSet& patchIDs
)
:
    patchDistMethod(mesh, patchIDs),
    coeffs_(dict.optionalSubDict(type() + "Coeffs")),

    // Wall distance from file
    yWall_to_load_
    (
        IOobject
        (
            "yWall_to_load",
            mesh.time().timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    ),

    // Normal vector from file
    nWall_to_load_
    (
        IOobject
        (
            "nWall_to_load",
            mesh.time().timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    )

    /**************************************************************************/

{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool Foam::patchDistMethods::Custom_externalImport::correct(volScalarField& y)
{
    return correct(y, const_cast<volVectorField&>(volVectorField::null()));
}


bool Foam::patchDistMethods::Custom_externalImport::correct
(
    volScalarField& y,
    volVectorField& n
)
{

    y = yWall_to_load_;

    // Only load n if the field is defined
    if (notNull(n))
    {
        n = nWall_to_load_;
    }

    return true;
}


// ************************************************************************* //

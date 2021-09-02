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

Modified by 
    Diego Hayashi Alonso (2020-2021)

Based on
    'OpenFOAM-7/src/TurbulenceModels/turbulenceModels/RAS/SpalartAllmaras/SpalartAllmaras.H'

\*---------------------------------------------------------------------------*/

#include "CustomSpalartAllmaras.H"
#include "fvOptions.H"
#include "bound.H"
#include "wallDist.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
tmp<volScalarField> CustomSpalartAllmaras<BasicTurbulenceModel>::chi() const
{
    // χ = ν_{T,aux}/ν
    return nuTilda_/this->nu();
}


template<class BasicTurbulenceModel>
tmp<volScalarField> CustomSpalartAllmaras<BasicTurbulenceModel>::fv1
(
    const volScalarField& chi
) const
{
    // χ³
    const volScalarField chi3(pow3(chi));

    // f_v1 = χ³/(χ³ + C_v1³)
    return chi3/(chi3 + pow3(Cv1_));
}


template<class BasicTurbulenceModel>
tmp<volScalarField> CustomSpalartAllmaras<BasicTurbulenceModel>::fv2
(
    const volScalarField& chi,
    const volScalarField& fv1
) const
{
    // f_v2 = 1 - χ/(1 + χ.f_v1)
    return 1.0 - chi/(1.0 + chi*fv1);
}


template<class BasicTurbulenceModel>
tmp<volScalarField> CustomSpalartAllmaras<BasicTurbulenceModel>::Stilda
(
    const volScalarField& chi,
    const volScalarField& fv1
) const
{

    // Ω = √2.||skew(∇U)||
       // skew(∇U) = 1/2 . (∇U - ∇U^T)
    volScalarField Omega(::sqrt(2.0)*mag(skew(fvc::grad(this->U_))));

    // S_tilda = max[Ω + f_v2(χ, f_v1).ν_{T,aux}/(κ.y)², C_s . Ω]
    return
    (
        max
        (
            Omega
          + fv2(chi, fv1)*nuTilda_/sqr(kappa_*y_),
            Cs_*Omega
        )
    );
}


template<class BasicTurbulenceModel>
tmp<volScalarField> CustomSpalartAllmaras<BasicTurbulenceModel>::fw
(
    const volScalarField& Stilda
) const
{
    // r = min(ν_{T,aux}/(max(S_tilda, ϵ_small).(κ.y)²), 10)
	// small = 1E-6 -> https://www.cfd-online.com/Forums/openfoam-solving/120990-small-great-rootvsmall-what.html
    volScalarField r
    (
        min
        (
            nuTilda_
           /(
               max
               (
                   Stilda,
                   dimensionedScalar(Stilda.dimensions(), small)
               )
              *sqr(kappa_*y_)
            ),
            scalar(10.0)
        )
    );
    // r = 0 on boundaries
    r.boundaryFieldRef() == 0.0;

    // g = r + c_w2.(r⁶ - r)
    const volScalarField g(r + Cw2_*(pow6(r) - r));

    // f_w = g.((1 + C_w3⁶)/(g⁶ + C_w3⁶))^{1/6}
    return g*pow((1.0 + pow6(Cw3_))/(pow6(g) + pow6(Cw3_)), 1.0/6.0);
}


template<class BasicTurbulenceModel>
void CustomSpalartAllmaras<BasicTurbulenceModel>::correctNut
(
    const volScalarField& fv1
)
{
    // ν_T = ν_{T, aux} . f_v1
    this->nut_ = nuTilda_*fv1;
    this->nut_.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(this->nut_);

    BasicTurbulenceModel::correctNut();
}


template<class BasicTurbulenceModel>
void CustomSpalartAllmaras<BasicTurbulenceModel>::correctNut()
{
    correctNut(fv1(this->chi()));
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
CustomSpalartAllmaras<BasicTurbulenceModel>::CustomSpalartAllmaras
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    eddyViscosity<RASModel<BasicTurbulenceModel>>
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),

    sigmaNut_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "sigmaNut",
            this->coeffDict_,
            0.66666
        )
    ),
    kappa_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "kappa",
            this->coeffDict_,
            0.41
        )
    ),

    Cb1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Cb1",
            this->coeffDict_,
            0.1355
        )
    ),
    Cb2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Cb2",
            this->coeffDict_,
            0.622
        )
    ),
    Cw1_(Cb1_/sqr(kappa_) + (1.0 + Cb2_)/sigmaNut_), // C_b1/κ²  + (1 + C_b2)/σ_{ν_{T,aux}}
    Cw2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Cw2",
            this->coeffDict_,
            0.3
        )
    ),
    Cw3_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Cw3",
            this->coeffDict_,
            2.0
        )
    ),
    Cv1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Cv1",
            this->coeffDict_,
            7.1
        )
    ),
    Cs_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Cs",
            this->coeffDict_,
            0.3
        )
    ),

    nuTilda_
    (
        IOobject
        (
            "nuTilda",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),

    y_(wallDist::New(this->mesh_).y()),

    /****************************** Material model ****************************/

    // Create dictionary of material model properties from I/O (Input/Ouput)
    materialmodelProperties (
    	IOobject (
    		"materialmodelProperties",
    		this->runTime_.constant(),
    		this->mesh_,
    		IOobject::MUST_READ_IF_MODIFIED,
    		IOobject::NO_WRITE
    	)
    ),

    // Design variable
    alpha_design_
    (
        IOobject
        (
            "alpha_design",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),

    // λ_v
    lambda_v_design_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "lambda_v_design",
            this->coeffDict_,
            1.0
        )
    ),

    // κ_min
     // https://www.openfoam.com/documentation/guides/latest/api/classFoam_1_1dimensioned.html#a5f0a14a277024329441b386b21d69c47
    k_min_
    (
            "k_min",
            dimensionSet(1, -3, -1, 0, 0, 0, 0),
            materialmodelProperties
    ),

    // κ_max
     // https://www.openfoam.com/documentation/guides/latest/api/classFoam_1_1dimensioned.html#a5f0a14a277024329441b386b21d69c47
    k_max_
    (
            "k_max",
            dimensionSet(1, -3, -1, 0, 0, 0, 0),
            materialmodelProperties
    ),

    // rho_density
     // https://www.openfoam.com/documentation/guides/latest/api/classFoam_1_1dimensioned.html#a5f0a14a277024329441b386b21d69c47
    rho_density_
    (
            "rho_density",
            dimensionSet(1, -3, 0, 0, 0, 0, 0),
            materialmodelProperties
    ),

    // q_penalization
     // https://www.openfoam.com/documentation/guides/latest/api/classFoam_1_1dimensioned.html#a5f0a14a277024329441b386b21d69c47
    q_penalization_
    (
            "q_penalization",
            dimless,
            materialmodelProperties
    )

    /**************************************************************************/

{
    if (type == typeName)
    {
        this->printCoeffs(type);
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool CustomSpalartAllmaras<BasicTurbulenceModel>::read()
{
    if (eddyViscosity<RASModel<BasicTurbulenceModel>>::read())
    {
        sigmaNut_.readIfPresent(this->coeffDict());
        kappa_.readIfPresent(this->coeffDict());

        Cb1_.readIfPresent(this->coeffDict());
        Cb2_.readIfPresent(this->coeffDict());
        Cw1_ = Cb1_/sqr(kappa_) + (1.0 + Cb2_)/sigmaNut_; // C_b1/κ²  + (1 + C_b2)/σ_{ν_{T,aux}}
        Cw2_.readIfPresent(this->coeffDict());
        Cw3_.readIfPresent(this->coeffDict());
        Cv1_.readIfPresent(this->coeffDict());
        Cs_.readIfPresent(this->coeffDict());

        return true;
    }
    else
    {
        return false;
    }
}


template<class BasicTurbulenceModel>
tmp<volScalarField> CustomSpalartAllmaras<BasicTurbulenceModel>::DnuTildaEff() const
{
    // D_{ν_{T,aux}} = (ν_{T,aux} + ν)/σ_{ν_{T,aux}}
    return volScalarField::New
    (
        "DnuTildaEff",
        (nuTilda_ + this->nu())/sigmaNut_
    );
}


template<class BasicTurbulenceModel>
tmp<volScalarField> CustomSpalartAllmaras<BasicTurbulenceModel>::k() const
{
    return volScalarField::New
    (
        "k",
        this->mesh_,
        dimensionedScalar(dimensionSet(0, 2, -2, 0, 0), 0)
    );
}


template<class BasicTurbulenceModel>
tmp<volScalarField> CustomSpalartAllmaras<BasicTurbulenceModel>::epsilon() const
{
    WarningInFunction
        << "Turbulence kinetic energy dissipation rate not defined for "
        << "Spalart-Allmaras model. Returning zero field"
        << endl;

    return volScalarField::New
    (
        "epsilon",
        this->mesh_,
        dimensionedScalar(dimensionSet(0, 2, -3, 0, 0), 0)
    );
}


template<class BasicTurbulenceModel>
void CustomSpalartAllmaras<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    // Local references


    // α_phase
    const alphaField& alpha = this->alpha_;

    // ρ
    const rhoField& rho = this->rho_;

    // α_phase.ρ.U = α_phase.ρ.φ
        // * φ = U
    const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;

    // fvOptions
    fv::options& fvOptions(fv::options::New(this->mesh_));

    eddyViscosity<RASModel<BasicTurbulenceModel>>::correct();

    // Compute χ
    const volScalarField chi(this->chi());

    // Compute f_v1
    const volScalarField fv1(this->fv1(chi));

    // Compute S_tilda
    const volScalarField Stilda(this->Stilda(chi, fv1));

    // Inverse permeability term
    volScalarField k_alpha(k_max_ + (k_min_ - k_max_)*(alpha_design_*(1 + q_penalization_)/(alpha_design_ + q_penalization_)));

    tmp<fvScalarMatrix> nuTildaEqn
    (

     // Time dependence
        // ∂(α_phase.ρ.ν_{T,aux})/∂t
        fvm::ddt(alpha, rho, nuTilda_)

     // Convection
        // + U•∇(α_phase.ρ.ν_{T,aux}) = ∇•(α_phase.ρ.U.ν_{T,aux}) (conservative form)
        // = + ∇•(α_phase.ρ.φ.ν_{T,aux})
           // * φ = U
      + fvm::div(alphaRhoPhi, nuTilda_)

      // Conservative diffusion term
        // - ∇•( (α_phase.ρ.D_{ν_{T,aux}}) . ∇ν_{T,aux} )
            // https://www.openfoam.com/documentation/guides/latest/doc/guide-schemes-laplacian-implementation-details.html
      - fvm::laplacian(alpha*rho*DnuTildaEff(), nuTilda_)
		// D_{ν_{T,aux})} = (ν_{T,aux} + ν)/σ_{ν_{T,aux})}

      // Non-conservative diffusion term
        // - C_b2/σ_{ν_{T,aux})}.ρ.|∇ν_{T,aux}|² 
      - Cb2_/sigmaNut_*alpha*rho*magSqr(fvc::grad(nuTilda_))

     ==

      // Production
        // + C_b1.α_phase.ρ.S_tilda.ν_{T,aux}
        Cb1_*alpha*rho*Stilda*nuTilda_

      // Destruction
        // - C_w1.α_phase.ρ.f_w(S_tilda).ν_{T,aux}/y² . ν_{T,aux}
      - fvm::Sp(Cw1_*alpha*rho*fw(Stilda)*nuTilda_/sqr(y_), nuTilda_)

      // Porous medium (for topology optimization)
        // - α_phase.ρ.λ_v.κ(α)/ρ_density . ν_{T,aux}
      - fvm::Sp(alpha*rho*lambda_v_design_*k_alpha/rho_density_, nuTilda_)

      // + fvOptions
      + fvOptions(alpha, rho, nuTilda_)
    );

    // Apply relaxation to ν_{T,aux}
    nuTildaEqn.ref().relax();

    // Constrain the values of ν_{T,aux} from fvOptions
    fvOptions.constrain(nuTildaEqn.ref());

    // Solve the equations for ν_{T,aux}
    solve(nuTildaEqn);

    // Correct the values of ν_{T,aux} from fvOptions
    fvOptions.correct(nuTilda_);

    // Bound the values of ν_{T,aux} to be higher than zero
    bound(nuTilda_, dimensionedScalar(nuTilda_.dimensions(), 0));

    // Correct the boundary conditions with the computed ν_{T,aux}
    nuTilda_.correctBoundaryConditions();

    // Correct the value of the turbulent viscosity ν_T
    correctNut(fv1);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// ************************************************************************* //

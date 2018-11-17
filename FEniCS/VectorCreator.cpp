#include <gtest.hh>

#include <dolfin.h>

#include <Spacy/Spacy.h>
#include <Spacy/Adapter/FEniCS/VectorSpace.h>

#include "LinearHeat.h"
#include "L2Functional.h"


using namespace Spacy;

// test setup
const constexpr auto scalar_degrees_of_freedom = 2;
const constexpr auto degrees_of_freedom = scalar_degrees_of_freedom * scalar_degrees_of_freedom;
const constexpr auto number_of_variables = 3;

const auto mesh1D = std::make_shared<dolfin::UnitIntervalMesh>(MPI_COMM_WORLD, degrees_of_freedom-1);
const auto mesh2D = std::make_shared<dolfin::UnitSquareMesh>(scalar_degrees_of_freedom-1, scalar_degrees_of_freedom-1);
const auto dolfin_V1D = std::make_shared<LinearHeat::FunctionSpace>(mesh1D);
const auto dolfin_V2D = std::make_shared<L2Functional::CoefficientSpace_x>(mesh2D);
const auto V1D = Spacy::FEniCS::makeHilbertSpace(dolfin_V1D);
const auto V2D = Spacy::FEniCS::makeHilbertSpace(dolfin_V2D, {0,1,2}, {});
const auto V2D_perm = Spacy::FEniCS::makeHilbertSpace(dolfin_V2D, {2,0,1}, {});
const auto V2DPrimalDual = Spacy::FEniCS::makeHilbertSpace(dolfin_V2D, {0,1}, {2});


TEST(FEniCS,SingleSpaceCreator)
{
    const auto& V = creator<FEniCS::VectorCreator>(V1D);
    EXPECT_EQ( V.size(), degrees_of_freedom );
    EXPECT_EQ( V.dofmap(0), 0 );
    EXPECT_EQ( V.dofmap(1), 1 );
    EXPECT_EQ( V.inverseDofmap(0), 0 );
    EXPECT_EQ( V.inverseDofmap(1), 1 );
}

TEST(FEniCS,ProductSpaceCreator)
{
    const auto& X = creator<ProductSpace::VectorCreator>(V2D);
    const auto& Y = creator<FEniCS::VectorCreator>(X.subSpace(0));
    const auto& U = creator<FEniCS::VectorCreator>(X.subSpace(1));
    const auto& P = creator<FEniCS::VectorCreator>(X.subSpace(2));

    EXPECT_EQ( Y.size(), degrees_of_freedom );
    EXPECT_EQ( U.size(), degrees_of_freedom );
    EXPECT_EQ( P.size(), degrees_of_freedom );
}

TEST(FEniCS,ProductSpaceCreator_IdMap)
{
    const auto& X = creator<ProductSpace::VectorCreator>(V2D);
    EXPECT_EQ( X.idMap(0), 0 );
    EXPECT_EQ( X.idMap(1), 1 );
    EXPECT_EQ( X.idMap(2), 2 );

    EXPECT_EQ( X.inverseIdMap(0), 0 );
    EXPECT_EQ( X.inverseIdMap(1), 1 );
    EXPECT_EQ( X.inverseIdMap(2), 2 );


    const auto& X_perm = creator<ProductSpace::VectorCreator>(V2D_perm);
    EXPECT_EQ( X_perm.idMap(0), 1 );
    EXPECT_EQ( X_perm.idMap(1), 2 );
    EXPECT_EQ( X_perm.idMap(2), 0 );

    EXPECT_EQ( X_perm.inverseIdMap(0), 2 );
    EXPECT_EQ( X_perm.inverseIdMap(1), 0 );
    EXPECT_EQ( X_perm.inverseIdMap(2), 1 );
}

TEST(FEniCS,ProductSpaceCreator_DofMap)
{
    const auto& X = creator<ProductSpace::VectorCreator>(V2D);
    const auto& Y = creator<FEniCS::VectorCreator>(X.subSpace(0));
    const auto& U = creator<FEniCS::VectorCreator>(X.subSpace(1));
    const auto& P = creator<FEniCS::VectorCreator>(X.subSpace(2));

    for(auto i=0; i<degrees_of_freedom; ++i)
    {
        EXPECT_EQ( Y.dofmap(i), i*number_of_variables );
        EXPECT_EQ( Y.inverseDofmap(i*number_of_variables), i );
    }
    for(auto i=0; i<degrees_of_freedom; ++i)
    {
        EXPECT_EQ( U.dofmap(i), i*number_of_variables+1 );
        EXPECT_EQ( U.inverseDofmap(i*number_of_variables+1), i );
    }
    for(auto i=0; i<degrees_of_freedom; ++i)
    {
        EXPECT_EQ( P.dofmap(i), i*number_of_variables+2 );
        EXPECT_EQ( P.inverseDofmap(i*number_of_variables+2), i );
    }
}

TEST(FEniCS,ProductSpaceCreator_DofMap_PermutedSpace)
{
    const auto& X = creator<ProductSpace::VectorCreator>(V2D_perm);
    const auto& Y = creator<FEniCS::VectorCreator>(X.subSpace(X.idMap(0)));
    const auto& U = creator<FEniCS::VectorCreator>(X.subSpace(X.idMap(1)));
    const auto& P = creator<FEniCS::VectorCreator>(X.subSpace(X.idMap(2)));

    for(auto i=0; i<degrees_of_freedom; ++i)
    {
        EXPECT_EQ( Y.dofmap(i), i*number_of_variables );
        EXPECT_EQ( Y.inverseDofmap(i*number_of_variables), i );
    }
    for(auto i=0; i<degrees_of_freedom; ++i)
    {
        EXPECT_EQ( U.dofmap(i), i*number_of_variables+1 );
        EXPECT_EQ( U.inverseDofmap(i*number_of_variables+1), i );
    }
    for(auto i=0; i<degrees_of_freedom; ++i)
    {
        EXPECT_EQ( P.dofmap(i), i*number_of_variables+2 );
        EXPECT_EQ( P.inverseDofmap(i*number_of_variables+2), i );
    }
}

TEST(FEniCS,PrimalDualProductSpaceCreator)
{
    const auto& X = creator<ProductSpace::VectorCreator>(V2DPrimalDual);
    const auto& X_primal = creator<ProductSpace::VectorCreator>(X.subSpace(0));
    const auto& X_dual = creator<ProductSpace::VectorCreator>(X.subSpace(1));
    const auto& Y = creator<FEniCS::VectorCreator>(X_primal.subSpace(0));
    const auto& U = creator<FEniCS::VectorCreator>(X_primal.subSpace(1));
    const auto& P = creator<FEniCS::VectorCreator>(X_dual.subSpace(0));

    EXPECT_EQ( Y.size(), degrees_of_freedom );
    EXPECT_EQ( U.size(), degrees_of_freedom );
    EXPECT_EQ( P.size(), degrees_of_freedom );
}

TEST(FEniCS,PrimalDualProductSpaceCreator_IdMap)
{
    const auto& X = creator<ProductSpace::VectorCreator>(V2DPrimalDual);
    const auto& X_primal = creator<ProductSpace::VectorCreator>(X.subSpace(0));
    const auto& X_dual = creator<ProductSpace::VectorCreator>(X.subSpace(1));

    EXPECT_EQ( X.idMap(0), 0 );
    EXPECT_EQ( X.idMap(1), 1 );

    EXPECT_EQ( X.inverseIdMap(0), 0 );
    EXPECT_EQ( X.inverseIdMap(1), 1 );

    EXPECT_EQ( X_primal.idMap(0), 0 );
    EXPECT_EQ( X_primal.idMap(1), 1 );
    EXPECT_EQ( X_dual.idMap(2), 0 );

    EXPECT_EQ( X_primal.inverseIdMap(0), 0 );
    EXPECT_EQ( X_primal.inverseIdMap(1), 1 );
    EXPECT_EQ( X_dual.inverseIdMap(0), 2 );
}


TEST(FEniCS,PrimalDualProductSpaceCreator_DofMap)
{
    const auto& X = creator<ProductSpace::VectorCreator>(V2DPrimalDual);
    const auto& X_primal = creator<ProductSpace::VectorCreator>(X.subSpace(0));
    const auto& X_dual = creator<ProductSpace::VectorCreator>(X.subSpace(1));
    const auto& Y = creator<FEniCS::VectorCreator>(X_primal.subSpace(0));
    const auto& U = creator<FEniCS::VectorCreator>(X_primal.subSpace(1));
    const auto& P = creator<FEniCS::VectorCreator>(X_dual.subSpace(0));

    for(auto i=0; i<degrees_of_freedom; ++i)
    {
        EXPECT_EQ( Y.dofmap(i), i*number_of_variables );
        EXPECT_EQ( Y.inverseDofmap(i*number_of_variables), i );
    }
    for(auto i=0; i<degrees_of_freedom; ++i)
    {
        EXPECT_EQ( U.dofmap(i), i*number_of_variables+1 );
        EXPECT_EQ( U.inverseDofmap(i*number_of_variables+1), i );
    }
    for(auto i=0; i<degrees_of_freedom; ++i)
    {
        EXPECT_EQ( P.dofmap(i), i*number_of_variables+2 );
        EXPECT_EQ( P.inverseDofmap(i*number_of_variables+2), i );
    }
}

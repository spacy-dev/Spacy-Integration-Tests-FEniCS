#include <gtest.hh>

#include <dolfin.h>

#include <Spacy/Spacy.h>
#include <Spacy/Adapter/FEniCS/Copy.h>
#include <Spacy/Adapter/FEniCS/Vector.h>
#include <Spacy/Adapter/FEniCS/VectorSpace.h>

#include "LinearHeat.h"
#include "L2Functional.h"

#include <iostream>

using namespace Spacy;

// test setup
constexpr int scalar_degrees_of_freedom = 2;
constexpr int degrees_of_freedom = scalar_degrees_of_freedom * scalar_degrees_of_freedom;
const auto mesh1D = std::make_shared<dolfin::UnitIntervalMesh>(MPI_COMM_WORLD, degrees_of_freedom-1);
const auto mesh2D = std::make_shared<dolfin::UnitSquareMesh>(scalar_degrees_of_freedom-1, scalar_degrees_of_freedom-1);
const auto dolfin_V1D = std::make_shared<LinearHeat::FunctionSpace>(mesh1D);
const auto dolfin_V2D = std::make_shared<L2Functional::CoefficientSpace_x>(mesh2D);
const auto V1D = Spacy::FEniCS::makeHilbertSpace(dolfin_V1D);
constexpr auto number_of_variables = 3;
const auto V2D = Spacy::FEniCS::makeHilbertSpace(dolfin_V2D, {0,1,2}, {});
const auto V2D_perm = Spacy::FEniCS::makeHilbertSpace(dolfin_V2D, {2,0,1}, {});
const auto V2DPrimalDual = Spacy::FEniCS::makeHilbertSpace(dolfin_V2D, {0,1}, {2});

namespace
{
    auto test_vector_1D()
    {
        auto v = zero(V1D);
        auto& v_ = cast_ref<FEniCS::Vector>(v).get();
        v_.setitem(0,1);
        v_.setitem(1,2);
        return v;
    }

    auto test_vector_2D()
    {
        auto v = zero(V2D);
        auto& v_ = cast_ref<ProductSpace::Vector>(v);
        auto& vy = cast_ref<FEniCS::Vector>(v_.component(0)).get();
        auto& vu = cast_ref<FEniCS::Vector>(v_.component(1)).get();
        auto& vp = cast_ref<FEniCS::Vector>(v_.component(2)).get();
        for(auto i=0; i<degrees_of_freedom; ++i)
        {
            vy.setitem(i,10+i);
            vu.setitem(i,100+i);
            vp.setitem(i,1000+i);
        }
        return v;
    }

    auto primal_dual_test_vector_2D()
    {
        auto v = zero(V2DPrimalDual);
        auto& v_ = cast_ref<ProductSpace::Vector>(v);
        auto& v_primal = cast_ref<ProductSpace::Vector>(v_.component(PRIMAL));
        auto& v_dual = cast_ref<ProductSpace::Vector>(v_.component(DUAL));

        auto& vy = cast_ref<FEniCS::Vector>(v_primal.component(0)).get();
        auto& vu = cast_ref<FEniCS::Vector>(v_primal.component(1)).get();
        auto& vp = cast_ref<FEniCS::Vector>(v_dual.component(0)).get();
        for(auto i=0; i<degrees_of_freedom; ++i)
        {
            vy.setitem(i,10+i);
            vu.setitem(i,100+i);
            vp.setitem(i,1000+i);
        }
        return v;
    }

    auto test_function_1D()
    {
        auto f = dolfin::Function(dolfin_V1D);
        f.vector()->setitem(0,2);
        f.vector()->setitem(1,3);
        return f;
    }

    auto test_function_2D()
    {
        auto f = dolfin::Function(dolfin_V2D);

        for(auto j=0; j<number_of_variables; ++j)
            for(auto i=0; i<degrees_of_freedom; ++i)
                f.vector()->setitem(i*number_of_variables +j, pow(10,j+1) + i );
        return f;
    }
}


// Copy normal vector
TEST(FEniCSUtilCopy,SpacyVectorToDolfinGenericVector_OneVariable)
{
    auto v = test_vector_1D();
    auto f = dolfin::Function(dolfin_V1D);

    FEniCS::copy(v, *f.vector());

    EXPECT_EQ( (*f.vector())[0], 1 );
    EXPECT_EQ( (*f.vector())[1], 2 );
}

TEST(FEniCSUtilCopy,DolfinGenericVectorToSpacyVector_OneVariable)
{
    auto v = zero(V1D);
    auto f = test_function_1D();

    FEniCS::copy(*f.vector(), v);

    const auto& v_ = cast_ref<FEniCS::Vector>(v).get();
    EXPECT_EQ( v_[0], 2 );
    EXPECT_EQ( v_[1], 3 );
}

TEST(FEniCSUtilCopy,SpacyVectorToDolfinFunction_OneVariable)
{
    auto v = test_vector_1D();
    auto f = dolfin::Function(dolfin_V1D);

    FEniCS::copy(v, f);

    EXPECT_EQ( (*f.vector())[0], 1 );
    EXPECT_EQ( (*f.vector())[1], 2 );
}

TEST(FEniCSUtilCopy,DolfinFunctionToSpacyVector_OneVariable)
{
    auto v = zero(V1D);
    auto f = test_function_1D();

    FEniCS::copy(f, v);

    const auto& v_ = cast_ref<FEniCS::Vector>(v).get();
    EXPECT_EQ( v_[0], 2 );
    EXPECT_EQ( v_[1], 3 );
}


// Copy product space vector
TEST(FEniCSUtilCopy,SpacyVectorToDolfinGenericVector_ProductSpace_ThreeVariables)
{
    auto v = test_vector_2D();
    auto f = dolfin::Function(dolfin_V2D);

    FEniCS::copy(v, *f.vector());

    for(auto j=0; j<number_of_variables; ++j)
        for(auto i=0; i<degrees_of_freedom; ++i)
            EXPECT_EQ( (*f.vector())[i*number_of_variables +j], pow(10,j+1) + i );
}

TEST(FEniCSUtilCopy,DolfinGenericVectorToSpacyVector_ProductSpace_ThreeVariables)
{
    auto v = zero(V2D);
    auto f = test_function_2D();

    FEniCS::copy(*f.vector(), v);

    const auto& vp_ = cast_ref<ProductSpace::Vector>(v);
    const auto& vy = cast_ref<FEniCS::Vector>(vp_.component(0)).get();
    const auto& vu = cast_ref<FEniCS::Vector>(vp_.component(1)).get();
    const auto& vp = cast_ref<FEniCS::Vector>(vp_.component(2)).get();

    constexpr auto y_offset = 10;
    for(auto i=0; i<degrees_of_freedom; ++i)
        EXPECT_EQ( vy[i], y_offset + i );

    const auto u_offset = 100;
    for(auto i=0; i<degrees_of_freedom; ++i)
        EXPECT_EQ( vu[i], u_offset + i );

    const auto p_offset = 1000;
    for(auto i=0; i<degrees_of_freedom; ++i)
        EXPECT_EQ( vp[i], p_offset + i );
}

TEST(FEniCSUtilCopy,SpacyVectorToDolfinFunction_ProductSpace_ThreeVariables)
{
    auto v = test_vector_2D();
    auto f = dolfin::Function(dolfin_V2D);

    FEniCS::copy(v, f);

    for(auto j=0; j<number_of_variables; ++j)
        for(auto i=0; i<degrees_of_freedom; ++i)
            EXPECT_EQ( (*f.vector())[i*number_of_variables +j], pow(10,j+1) + i );
}

TEST(FEniCSUtilCopy,DolfinFunctionToSpacyVector_ProductSpace_ThreeVariables)
{
    auto v = zero(V2D);
    auto f = test_function_2D();

    FEniCS::copy(f, v);

    const auto& vp_ = cast_ref<ProductSpace::Vector>(v);
    const auto& vy = cast_ref<FEniCS::Vector>(vp_.component(0)).get();
    const auto& vu = cast_ref<FEniCS::Vector>(vp_.component(1)).get();
    const auto& vp = cast_ref<FEniCS::Vector>(vp_.component(2)).get();

    constexpr auto
            y_offset = 10,
            u_offset = 100,
            p_offset = 1000;
    for(auto i=0; i<degrees_of_freedom; ++i)
    {
        EXPECT_EQ( vy[i], y_offset + i );
        EXPECT_EQ( vu[i], u_offset + i );
        EXPECT_EQ( vp[i], p_offset + i );
    }
}


// Copy permuted product space vector
TEST(FEniCSUtilCopy,SpacyVectorToDolfinGenericVector_PermutedProductSpace_ThreeVariables)
{
    auto v = test_vector_2D();
    auto f = dolfin::Function(dolfin_V2D);

    FEniCS::copy(v, *f.vector());

    for(auto j=0; j<number_of_variables; ++j)
        for(auto i=0; i<degrees_of_freedom; ++i)
            EXPECT_EQ( (*f.vector())[i*number_of_variables +j], pow(10,j+1) + i );
}

TEST(FEniCSUtilCopy,DolfinGenericVectorToSpacyVector_PermutedProductSpace_ThreeVariables)
{
    auto v = zero(V2D_perm);
    auto f = test_function_2D();

    FEniCS::copy(*f.vector(), v);

    const auto& vp_ = cast_ref<ProductSpace::Vector>(v);
    const auto& vy = cast_ref<FEniCS::Vector>(vp_.component(1)).get();
    const auto& vu = cast_ref<FEniCS::Vector>(vp_.component(2)).get();
    const auto& vp = cast_ref<FEniCS::Vector>(vp_.component(0)).get();

    constexpr auto
            y_offset = 10,
            u_offset = 100,
            p_offset = 1000;
    for(auto i=0; i<degrees_of_freedom; ++i)
    {
        EXPECT_EQ( vy[i], y_offset + i );
        EXPECT_EQ( vu[i], u_offset + i );
        EXPECT_EQ( vp[i], p_offset + i );
    }
}

TEST(FEniCSUtilCopy,SpacyVectorToDolfinFunction_PermutedProductSpace_ThreeVariables)
{
    auto v = test_vector_2D();
    auto f = dolfin::Function(dolfin_V2D);

    FEniCS::copy(v, f);

    for(auto j=0; j<number_of_variables; ++j)
        for(auto i=0; i<degrees_of_freedom; ++i)
            EXPECT_EQ( (*f.vector())[i*number_of_variables +j], pow(10,j+1) + i );
}

TEST(FEniCSUtilCopy,DolfinFunctionToSpacyVector_PermutedProductSpace_ThreeVariables)
{
    auto v = zero(V2D_perm);
    auto f = test_function_2D();

    FEniCS::copy(f, v);

    const auto& vp_ = cast_ref<ProductSpace::Vector>(v);
    const auto& vy = cast_ref<FEniCS::Vector>(vp_.component(1)).get();
    const auto& vu = cast_ref<FEniCS::Vector>(vp_.component(2)).get();
    const auto& vp = cast_ref<FEniCS::Vector>(vp_.component(0)).get();

    constexpr auto
            y_offset = 10,
            u_offset = 100,
            p_offset = 1000;
    for(auto i=0; i<degrees_of_freedom; ++i)
    {
        EXPECT_EQ( vy[i], y_offset + i );
        EXPECT_EQ( vu[i], u_offset + i );
        EXPECT_EQ( vp[i], p_offset + i );
    }
}


// Copy primal dualproduct space vector
TEST(FEniCSUtilCopy,SpacyVectorToDolfinGenericVector_PrimalDualProductSpace_ThreeVariables)
{
    auto v = primal_dual_test_vector_2D();
    auto f = dolfin::Function(dolfin_V2D);

    FEniCS::copy(v, *f.vector());

    for(auto j=0; j<number_of_variables; ++j)
        for(auto i=0; i<degrees_of_freedom; ++i)
            EXPECT_EQ( (*f.vector())[i*number_of_variables +j], pow(10,j+1) + i );
}

TEST(FEniCSUtilCopy,DolfinGenericVectorToSpacyVector_PrimalDualProductSpace_ThreeVariables)
{
    auto v = zero(V2DPrimalDual);
    auto f = test_function_2D();

    FEniCS::copy(*f.vector(), v);

    ASSERT_TRUE(is<ProductSpace::Vector>(v));
    const auto& v_ = cast_ref<ProductSpace::Vector>(v);

    ASSERT_TRUE(is<ProductSpace::Vector>(v_.component(PRIMAL)));
    const auto& vp_ = cast_ref<ProductSpace::Vector>(v_.component(PRIMAL));
    ASSERT_TRUE(is<ProductSpace::Vector>(v_.component(DUAL)));
    const auto& vd_ = cast_ref<ProductSpace::Vector>(v_.component(DUAL));

    ASSERT_TRUE(is<FEniCS::Vector>(vp_.component(0)));
    const auto& vy = cast_ref<FEniCS::Vector>(vp_.component(0)).get();
    ASSERT_TRUE(is<FEniCS::Vector>(vp_.component(1)));
    const auto& vu = cast_ref<FEniCS::Vector>(vp_.component(1)).get();
    ASSERT_TRUE(is<FEniCS::Vector>(vd_.component(0)));
    const auto& vp = cast_ref<FEniCS::Vector>(vd_.component(0)).get();

    constexpr auto y_offset = 10;
    for(auto i=0; i<degrees_of_freedom; ++i)
        EXPECT_EQ( vy[i], y_offset + i );

    const auto u_offset = 100;
    for(auto i=0; i<degrees_of_freedom; ++i)
        EXPECT_EQ( vu[i], u_offset + i );

    const auto p_offset = 1000;
    for(auto i=0; i<degrees_of_freedom; ++i)
        EXPECT_EQ( vp[i], p_offset + i );
}

TEST(FEniCSUtilCopy,SpacyVectorToDolfinFunction_PrimalDualProductSpace_ThreeVariables)
{
    auto v = primal_dual_test_vector_2D();
    auto f = dolfin::Function(dolfin_V2D);

    FEniCS::copy(v, f);

    for(auto j=0; j<number_of_variables; ++j)
        for(auto i=0; i<degrees_of_freedom; ++i)
            EXPECT_EQ( (*f.vector())[i*number_of_variables +j], pow(10,j+1) + i );
}

TEST(FEniCSUtilCopy,DolfinFunctionToSpacyVector_PrimalDualProductSpace_ThreeVariables)
{
    auto v = zero(V2DPrimalDual);
    auto f = test_function_2D();

    FEniCS::copy(f, v);

    ASSERT_TRUE(is<ProductSpace::Vector>(v));
    const auto& v_ = cast_ref<ProductSpace::Vector>(v);

    ASSERT_TRUE(is<ProductSpace::Vector>(v_.component(PRIMAL)));
    const auto& vp_ = cast_ref<ProductSpace::Vector>(v_.component(PRIMAL));
    ASSERT_TRUE(is<ProductSpace::Vector>(v_.component(DUAL)));
    const auto& vd_ = cast_ref<ProductSpace::Vector>(v_.component(DUAL));

    ASSERT_TRUE(is<FEniCS::Vector>(vp_.component(0)));
    const auto& vy = cast_ref<FEniCS::Vector>(vp_.component(0)).get();
    ASSERT_TRUE(is<FEniCS::Vector>(vp_.component(1)));
    const auto& vu = cast_ref<FEniCS::Vector>(vp_.component(1)).get();
    ASSERT_TRUE(is<FEniCS::Vector>(vd_.component(0)));
    const auto& vp = cast_ref<FEniCS::Vector>(vd_.component(0)).get();

    constexpr auto y_offset = 10;
    for(auto i=0; i<degrees_of_freedom; ++i)
        EXPECT_EQ( vy[i], y_offset + i );

    const auto u_offset = 100;
    for(auto i=0; i<degrees_of_freedom; ++i)
        EXPECT_EQ( vu[i], u_offset + i );

    const auto p_offset = 1000;
    for(auto i=0; i<degrees_of_freedom; ++i)
        EXPECT_EQ( vp[i], p_offset + i );
}

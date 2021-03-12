#include <gtest.hh>

#include <dolfin.h>

#include <Spacy/Spacy.h>
#include <Spacy/Adapter/FEniCS/Vector.h>
#include <Spacy/Adapter/FEniCS/VectorSpace.h>

#include "LinearHeat.h"

using namespace Spacy;

namespace
{
    auto get_test_vector(const Spacy::VectorSpace& V, int degrees_of_freedom)
    {
        auto v = dolfin::Function(Spacy::creator<FEniCS::VectorCreator>(V).get());
        for(auto i=0; i<degrees_of_freedom; ++i)
            v.vector()->setitem(i, i);
        v.vector()->apply("insert");
        return v;
    }

    const int degrees_of_freedom = 2;
    const auto mesh = std::make_shared<dolfin::UnitIntervalMesh>(MPI_COMM_WORLD, degrees_of_freedom-1);
    const auto dolfin_V = std::make_shared<LinearHeat::FunctionSpace>(mesh);
    auto V = Spacy::FEniCS::makeHilbertSpace(dolfin_V);
}

TEST(FEniCSVectorAdapter,CreateFromFEniCSFunction)
{
    auto v = get_test_vector(V, degrees_of_freedom);

    Spacy::FEniCS::Vector w(v,V);

    EXPECT_EQ( w.get()[0] , 0.);
    EXPECT_EQ( w.get()[1] , 1.);
}

TEST(FEniCSVectorAdapter,AssignFromFEniCSFunction)
{
    auto v = get_test_vector(V, degrees_of_freedom);

    auto w0 = Spacy::FEniCS::Vector(V);
    auto w = Spacy::FEniCS::Vector(v,V);

    EXPECT_EQ( w.get()[0] , 0.);
    EXPECT_EQ( w.get()[1] , 1.);
    EXPECT_EQ( w0.get()[0] , 0.);
    EXPECT_EQ( w0.get()[1] , 0.);

    w = v;
    w0 = v;

    EXPECT_EQ( w.get()[0] , 0.);
    EXPECT_EQ( w.get()[1] , 1.);
    EXPECT_EQ( w0.get()[0] , 0.);
    EXPECT_EQ( w0.get()[1] , 1.);
}

TEST(FEniCSVectorAdapter,AddAssign)
{
    auto v = get_test_vector(V, degrees_of_freedom);

    Spacy::FEniCS::Vector w(v,V);
    Spacy::FEniCS::Vector w0(v,V);
    w += w0;

    EXPECT_EQ( w.get()[0] , 0.);
    EXPECT_EQ( w.get()[1] , 2.);

    Spacy::FEniCS::Vector w1(V);

    w1 += w0;

    EXPECT_EQ( w1.get()[0] , 0.);
    EXPECT_EQ( w1.get()[1] , 1.);
}

TEST(FEniCSVectorAdapter,SubtractAssign)
{
    auto v = get_test_vector(V, degrees_of_freedom);

    Spacy::FEniCS::Vector w(V);
    Spacy::FEniCS::Vector w0(v,V);
    w -= w0;

    EXPECT_EQ( w.get()[0] , 0.);
    EXPECT_EQ( w.get()[1] , -1.);
}

TEST(FEniCSVectorAdapter,MultiplyWithScalar)
{
    auto v = get_test_vector(V, degrees_of_freedom);

    Spacy::FEniCS::Vector w(v,V);
    w *= 2;

    EXPECT_EQ( w.get()[0] , 0.);
    EXPECT_EQ( w.get()[1] , 2.);
}

TEST(FEniCSVectorAdapter,ApplyAsDual)
{
    auto v = get_test_vector(V, degrees_of_freedom);

    Spacy::FEniCS::Vector w(v,V);
    auto dp = w(w);

    EXPECT_EQ(get(dp), 1.);

    w *= 2;
    dp = w(w);

    EXPECT_EQ(get(dp), 4.);
}

TEST(FEniCSVectorAdapter,Negation)
{
    auto v = get_test_vector(V, degrees_of_freedom);

    Spacy::FEniCS::Vector w(v,V);
    auto w1 = -w;

    EXPECT_EQ( w1.get()[0] , 0.);
    EXPECT_EQ( w1.get()[1] , -1.);
}

TEST(FEniCSVectorAdapter,Comparison)
{
    auto v = get_test_vector(V, degrees_of_freedom);

    Spacy::FEniCS::Vector w0(v,V);
    Spacy::FEniCS::Vector w1(v,V);

    EXPECT_TRUE( w0 == w1 );

    const auto eps = 1e-5;
    V.setEps(eps);

    w0.get().setitem(1, 1 - 0.5*eps);
    EXPECT_TRUE( w0 == w1 );

    w0.get().setitem(1, 1 - 1.1*eps);
    EXPECT_FALSE( w0 == w1 );
}

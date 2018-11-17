#pragma once

#include <Spacy/Vector.h>
#include <Spacy/Spaces/RealSpace.h>

namespace Mock
{
  struct Norm
  {
    static constexpr int testValue = 5;
    Spacy::Real operator()(const ::Spacy::Vector&) const;
  };

  struct Norm10
  {
    static constexpr int testValue = 10;
    Spacy::Real operator()(const ::Spacy::Vector&) const;
  };
}

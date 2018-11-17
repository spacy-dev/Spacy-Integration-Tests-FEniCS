#include "VectorCreator.h"

#include <Spacy/Vector.h>
#include <Spacy/VectorSpace.h>

#include "Vector.h"

Mock::Vector Mock::VectorCreator::operator()(const Spacy::VectorSpace* space) const
{
  return Mock::Vector{*space};
}

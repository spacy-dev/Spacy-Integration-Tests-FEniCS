#include "Norm.h"

#include <Spacy/Operator.h>

Spacy::Real Mock::Norm::operator()(const ::Spacy::Vector&) const
{
  return Spacy::Real(testValue);
}

Spacy::Real Mock::Norm10::operator()(const ::Spacy::Vector&) const
{
  return Spacy::Real(testValue);
}


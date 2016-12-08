#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ufl import *
set_level(DEBUG)
# Compile this form with FFC: ffc -l dolfin L2Functional.ufl

Y = FiniteElement("Lagrange", line, 1)
U = FiniteElement("Lagrange", line, 1)
P = FiniteElement("Lagrange", line, 1)

X = MixedElement( Y , U , P )

x = Coefficient(X)
(y,u,p) = split(x)

F  = y*y*dx + u*u*dx + p*p*dx


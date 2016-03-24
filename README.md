# 3D_polarized_traces

Polarized traces preconditioner for the 3D Helmholtz equation, discretized using finite differences plus absorbing boundary conditions realized via PML's.

This is an very rough prototype. Its main purpouse is to show how to design transmission conditions via a discrete Green's representation formula. This version of the code is matrix free and it does not use compressed linear algebra.

The notation was mostly taken from Zepeda-Núñez and Demanet, "The method of polarized traces for the 2D Helmholtz equation, JCP 2015"

This code was mainly developed under Julia 0.3, it will work for newer versions but you may have a large amount of warnings.

You will need the package IterativeSolvers in order to run the examples.

We thank TOTAL for the genereous support.

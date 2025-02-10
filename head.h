#include <iostream>
#include <cmath>
#include "mkl.h"
#include "mpi.h"
#include <fstream>
#include <random>
#include <vector>
using std::ios;
using std::cout;
using std::endl;
using std::vector;

const double pi=3.141592653589793,
             e=1.602176487e-19,
             eps0=8.854187817e-12,
             hbar=1.054571628e-34;


const double G0[2][2]={{-sqrt(3.0)/2.0,-1.5},{sqrt(3.0),0.0}},
             K[2][2]={{-sqrt(3.0)/2.0,0.5},{-sqrt(3.0)/2.0,-0.5}},
             G1[4][2]={{sqrt(3.0)/2.0,1.5},{-sqrt(3.0)/2.0,-1.5},{-sqrt(3.0)/2.0,1.5},{sqrt(3.0)/2.0,-1.5}},
             a=1.42e-10,
             eps=7.0*eps0,
             BMcut=108.1;

const MKL_Complex16 alpha={1.0,0.0},
	            beta={0.0,0.0},
	            c0={0.0,0.0};

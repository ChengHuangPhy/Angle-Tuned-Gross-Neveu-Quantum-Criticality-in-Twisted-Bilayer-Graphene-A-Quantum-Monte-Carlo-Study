# Angle-Tuned-Gross-Neveu-Quantum-Criticality-in-Twisted-Bilayer-Graphene-A-Quantum-Monte-Carlo-Study
Code for the Monte Carlo simulation

System requirements: Ubuntu 24.04.1 LTS with Intel® oneAPI Math Kernel Library and Intel® MPI Library installed

To compile: mpiicpx -qmkl -std=c++11 cftbg.cpp function.o -o program. The compile time is usually several seconds.

To run the program: mpirun [...] ./program. The run time is usually several hours for L = 6 with one thousand sweeps.

Contol parameters are in the cftbg.cpp file.

Output: The first line contains the values of the imaginary time Green's function from all processers, and the total number of data is N_processor*(N_time+1)*N_k*2*2. One refers to the data as (((i_processor*(N_time+1)+i_time)*N_k+i_k)*2+i_band)*2+real/imag. The second line states the simulation tempareture. The third and following lines are for the average values and errors of imaginary time Green's function over all processors, with the total number of the lines is (N_time+1)*N_k*2. The refering is (i_time*N_k+i_k)*2+i_band. The last line states the IVC correlation and its error.

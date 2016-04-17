# Script to test the Helmholtz system in 3D using a direct method

# loading the functions needed to build the system
include("../src/HelmholtzMatrix.jl");

# number of deegres of freedom per dimension
# number of deegres of freedom per dimension
nx = 30;
ny = 30;
nz = 32;
npml = 6;

z = linspace(0,1,nz);


# the squared slowness, inthis case constant velocity
m = ones(nx,ny,nz);

# extra arguments
h     = z[2]-z[1];
fac   = 20/(npml*h);
order = 2;
K     = nx/10;
omega = 2*pi*K;

# bulding the matrix
print("Assembling the Hemholtz Matrix \n")
H = HelmholtzMatrix(m,nx,ny,nz,npml,h,fac,order,omega);

# factorizing the matrix
print("Factorizing Hemholtz Matrix \n")
@time C = lufact(H);

# defining the rhs geometrically
f = zeros(nx,ny,nz);
f[15,15,11] = 1000;

# total number of deegres of freedom
n = nx*ny*nz;

#solving the system by backsubstitution
@time uVec = C\f[:];

# reshaping the solution to physical space
u = reshape(uVec, (nx,ny,nz));

# defining the mkl solver  and testing with MKL Pardiso
using Pardiso

solverMKL = MKLPardisoSolver();

set_nprocs(solverMKL, 16)
set_mtype(solverMKL, 3)
set_iparm(solverMKL,12, 2)


#xx0 = Minv\bb
bb = zeros(Complex128,n) + f[:]
x0 = zeros(Complex128,n)
set_phase(solverMKL, 12)
@time pardiso(solverMKL, x0, H, bb)

set_iparm(solverMKL,12, 2)
set_phase(solverMKL, 33)
@time pardiso(solverMKL, x0, H, bb)


# trying to save the weavefield generated by the puntual source
using MAT
file = matopen("matfile301x301x21.mat", "w")
write(file, "u", u)
write(file, "f", f)
close(file)

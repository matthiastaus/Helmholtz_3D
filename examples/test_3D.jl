# Script to test the Helmholtz system in 3D using a direct method

# loading the functions needed to build the system
include("../src/HelmholtzMatrix.jl");

# number of deegres of freedom per dimension
# number of deegres of freedom per dimension
nx = 30;
ny = 30;
nz = 22;
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
C = lufact(H);

# defining the rhs geometrically
f = zeros(nx,ny,nz);
f[15,15,11] = 1000;

# total number of deegres of freedom
n = nx*ny*nz;


#solving the system by backsubstitution
tic();
uVec = C\f[:];
toc()

# reshaping the solution to physical space
u = reshape(uVec, (nx,ny,nz));

# plotting the solution
#using Winston
#ii = 15;
#imagesc(real(u[:,:,ii]),    (minimum(real(u[:,:,ii])), maximum(real(u[:,:,ii]))))

# trying to save the weavefield generated by the puntual source
using MAT
file = matopen("matfile301x301x21.mat", "w")
write(file, "u", u)
write(file, "f", f)
close(file)

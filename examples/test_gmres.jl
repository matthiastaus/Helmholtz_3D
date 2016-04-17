# Script to test the method of polarized traces to
# solve the 3D Helmholtz equation

# loading the functions needed to build the system
include("../src/subdomain2.jl");
include("../src/preconditioners.jl")


using IterativeSolvers

# number of deegres of freedom per dimension
nx = 40;
ny = 76;
nz = 76;
npml = 8;

# number of layers
nLayer = 6;

# interior degrees of freedom
# for simplicity we suppose that we have the same number of
# degrees of freedom at each layer
nzi = round(Int,(nz-2*npml)/nLayer);
# extended deegres of freedom within each layer
nzd = nzi+2*npml;

# we sample the z space, the solver is normalize such that the
# z legnth of the computational domain is always 1
z = linspace(0,1,nz);
zInd = [npml+1+nzi*(ii-1) for ii = 1:nLayer]


# extra arguments
h     = z[2]-z[1];    # mesh width
fac   = 20/(npml*h);  # absorbition coefficient for the PML
order = 2;            # order of the method, this should be changes
K     = nz/6;         # wave number
omega = 2*pi*K;       # frequency

#m = ones(nx,ny,nz);
# the squared slowness, we have a constant speed plus some bumps
m = zeros(Float64, nx,ny,nz);

# we define the model that we will use, in this case it is just a sum of 3
# Gaussian bumps
function bump(x,y,z,center, amplitude)
    return amplitude*( (x-center[1]).^2 + (y-center[2]).^2 + (x-center[3]).^2)
end

for ii=1:nx,jj=1:ny,kk=1:nz
    m[ii,jj,kk] = 1/( 1 + bump(ii*h,jj*h,kk*h, [0.5 0.5 0.5], 0.1 ) +
                          bump(ii*h,jj*h,kk*h, [0.2 0.3 0.7], 0.3 ) +
                          bump(ii*h,jj*h,kk*h, [0.2 0.3 0.7], 0.2 ) ).^2;
end


println("Solving the Helmholtz equation for nx = ",nx,", ny = ",ny, ", nz = ",nz,
        ", npml = ", npml, ", nLayer = ", nLayer, ", internal deegres of freedom = ", nzi,
        ", absorbtion factor = ", fac , " at frequency = ", omega);

# defining the full source (point source in this case)
n = nx*ny*nz;
f = zeros(Complex128,nx,ny,nz);
# due to a bug the source needs to be in the top layer
f[8,8,18] = n;


# bulding the domain decomposition data structure
println("Building the subdomains")
modelArray = [Model(m[:,:,(1:nzd)+nzi*(ii-1)], npml,collect(z),[0 0 z[1+npml+nzi*(ii-1)]],
			 h,fac,order,omega) for ii=1:nLayer];

# using Pardiso
# modelArray = [Model(m[:,:,(1:nzd)+nzi*(ii-1)], npml,collect(z),[0 0 z[1+npml+nzi*(ii-1)]],
#        h,fac,order,omega, solvertype  = "MKLPARDISO") for ii=1:nLayer];


#factorizing the local models
println("Factorizing the problems locally \n")
for ii = 1:nLayer
  factorize!(modelArray[ii])
  #convert64_32!(modelArray[ii]) #change the index basis to make MKLPardiso faster
end

subDomains = [Subdomain(modelArray[ii],1) for ii=1:nLayer];

#########################################################################
# Solving the local problem
# perform the local solves and extract the traces
uBdyPol = extractRHS(subDomains,f[:]);

# We vectorize the RHS, and put in the correct form
uBdyPer = -vectorizePolarizedBdyDataRHS(subDomains, uBdyPol)

##########################################################################
#  Solving for the boundary data

# allocating the preconditioner
Precond = IntegralPreconditioner(subDomains);

##############  GMRES #####################
# # solving for the traces

u = 0*uBdyPer;
@time data = gmres!(u,x->applyMMOpt2(subDomains,x), uBdyPer, Precond; tol=0.00001);

println("Number of iteration of GMRES : ", countnz( data[2].residuals[:]))

#########################################################################
# testing the solution

# we apply the polarized matrix to u to check for the error
MMu = applyMMOptUmf(subDomains, u);

println("Error for the polarized boundary integral system = ", norm(MMu - uBdyPer)/norm(uBdyPer) );

# adding the polarized traces to obtain the traces
# u = u^{\uparrow} + u^{\downarrow}
uBdySol = u[1:end/2]+u[(1+end/2):end];

uGamma = -vectorizeBdyData(subDomains, uBdyPol);

Mu = applyM(subDomains, uBdySol);

# checking that we recover the good solution
println("Error for the boundary integral system = ", norm(Mu - uGamma)/norm(uGamma));

########### Reconstruction ###############
# TODO this needs to be encapsulated too
# We still need to code the reconstruction that goes here

# Script to test the method of polarized traces to
# solve the 3D Helmholtz equation
# In this case this script is to test that the DDM
# encapsulation works as it should

# loading the functions needed to build the system
include("../src/subdomain2.jl");
include("../src/DomainDecomposition.jl")
include("../src/preconditioners.jl")


using IterativeSolvers

# options for the direct solver used within each layer
# Umfpack will be the slowest, and it is only for
# testing porpouses. You may want to install MUMPS
# or if you have a MKL license, use MKLPardiso
UmfpackBool    = false
MKLPardisoBool = true
MUMPSBool      = false

# loading the different solvers
MKLPardisoBool == true  && using Pardiso
MUMPSBool == true       && using MUMPS

# using some extra optimization to reduce the
# execution time
OptBool = false

# number of deegres of freedom per dimension
nx = 40;
ny = 50;
nz = 52;
npml = 6;

# the number of points have to such
# (nz - 2*npml)/nLayer is an integer

# number of layers
nLayer = 4;

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

# We place a point source
f[8,8,38] = n;


# bulding the domain decomposition data structure
println("Building the subdomains")

# The data structure is built depending on the local sparse direct solver that was
# selected and the profile for the PML, the default is a quadratic Profile
if UmfpackBool == true
  println("Using Umfpack as a sparse direct solver")
  modelArray = [Model(m[:,:,(1:nzd)+nzi*(ii-1)], npml,collect(z),[0 0 z[1+npml+nzi*(ii-1)]],
        h,fac,order,omega, (ii == 1)? "N": ((ii == nLayer)? "S": "M"), profileType="unbounded")
        for ii=1:nLayer];
end

if MKLPardisoBool == true
  # if PARDISO is installed it will load it
  println("Using MKL Pardiso as a sparse direct solver")
  modelArray = [Model(m[:,:,(1:nzd)+nzi*(ii-1)], npml,collect(z),[0 0 z[1+npml+nzi*(ii-1)]],
        h,fac,order,omega, (ii == 1)? "N": ((ii == nLayer)? "S": "M"), profileType="unbounded",
        solvertype  = "MKLPARDISO") for ii=1:nLayer];
end

if MUMPSBool == true
  # if MUMPS is installed it will load it
  println("Using MUMPS as a sparse direct solver")
  modelArray = [Model(m[:,:,(1:nzd)+nzi*(ii-1)], npml,collect(z),[0 0 z[1+npml+nzi*(ii-1)]],
       h,fac,order,omega,  (ii == 1)? "N": ((ii == nLayer)? "S": "M"), profileType="unbounded",
       solvertype = "MUMPS") for ii=1:nLayer];
end

#factorizing the local models
println("Factorizing the problems locally \n")
for ii = 1:nLayer
  factorize!(modelArray[ii])
  MKLPardisoBool && convert64_32!(modelArray[ii]) #change the index basis to make MKLPardiso faster
end

# creating an array of subdomains
subDomains = Array{Subdomain}(nLayer)
for ii = 1:nLayer
  subDomains[ii] = Subdomain(modelArray[ii],1)
end

DDM = DomainDecomposition(subDomains,1)

#########################################################################
# Solving the local problem
# perform the local solves and extract the traces
uBdyPol = extractRHS(subDomains,f[:]);

uBdyPolDDM = extractRHS(DDM,f[:]);


# We vectorize the RHS, and put in the correct form
uBdyPer = -vectorizePolarizedBdyDataRHS(subDomains, uBdyPol)

##########################################################################
#  Solving for the boundary data

# allocating the preconditioner
if OptBool == false
  # by default we use the Gauss-Seidel Preconditioner
  Precond = IntegralPreconditioner(subDomains);
else
  # we can use the optimized Gauss-Seidel that uses the
  # jump conditiones to perform one less local solve per layer
  Precond = IntegralPreconditioner(subDomains,precondtype ="GSOpt");
end

##############  GMRES #####################
# # solving for the traces

u = 0*uBdyPer;

@time data = gmres!(u,x->applyMMOptUmf(subDomains,x), uBdyPer, Precond; tol=0.00001);

println("Number of iteration of GMRES : ", countnz( data[2].residuals[:]));

##############  GMRES #####################
# # solving for the traces using the DDM encapsulation

PrecondDDM = IntegralPreconditionerDDM(DDM);


uDDM = 0*uBdyPer;

@time data = gmres!(uDDM,x->applyMMOptMultDMM(DDM,x), uBdyPer, PrecondDDM; tol=0.00001);

println("Number of iteration of GMRES : ", countnz( data[2].residuals[:]));


#########################################################################
# testing the solution

# we apply the polarized matrix to u to check for the error
@time MMu = applyMMOptUmf(subDomains, u);

@time MMuDDM = applyMMOptMultDMM(DDM, uDDM);

println("Error in the application of the integral operator is  = ", norm(MMu - MMuDDM) );

println("Error for the polarized boundary integral system = ", norm(MMu - uBdyPer)/norm(uBdyPer) );
println("Error for the polarized boundary integral system using DDM encapsulation = ", norm(MMuDDM - uBdyPer)/norm(uBdyPer) );



# adding the polarized traces to obtain the traces
# u = u^{\uparrow} + u^{\downarrow}
uBdySol = u[1:round(Integer,end/2)]+u[(1+round(Integer,end/2)):end];

uGamma = -vectorizeBdyData(subDomains, uBdyPol);

Mu = applyM(subDomains, uBdySol);

# checking that we recover the good solution
println("Error for the boundary integral system = ", norm(Mu - uGamma)/norm(uGamma));

########### Reconstruction ###############

(u0,u1,uN,uNp) = devectorizeBdyDataContiguous(subDomains, uBdySol);

uVol = reshape(reconstruction(subDomains, f, u0, u1, uN, uNp), nx, ny, nz);


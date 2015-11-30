# Script to test the method of polarized traces to
# solve the 2D Helmholtz equation

# loading the functions needed to build the system
include("../src/subdomain.jl");
include("../src/preconditioners.jl")
# number of deegres of freedom per dimension
nx = 40;
ny = 40;
nz = 62;
npml = 6;

# number of layers
nLayer = 5;

# interior degrees of freedom
nzi = int((nz-2*npml)/nLayer);
# extended deegres of freedom
nzd = nzi+2*npml;


z = linspace(0,1,nz);
zInd = {npml+1+nzi*(ii-1) for ii = 1:nLayer}


# extra arguments
h     = z[2]-z[1];
fac   = 20/(npml*h);
order = 2;
K     = nz/6;
omega = 2*pi*K;

#m = ones(nx,ny,nz);
# the squared slowness, we have a constant speed plus some bumps
m = zeros(Float64, nx,ny,nz);

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
f = zeros(nx,ny,nz);
f[20,20,15] = n;


# bulding the matrix
println("Building the subdomains and factorizing the problems locally \n")
subArray = { Subdomain(m[:,:,(1:nzd)+nzi*(ii-1)],npml,[0 0 z[1+npml+nzi*(ii-1)]],
			 h,fac,order,omega) for ii=1:nLayer};


####################################################################################
# Solving the local problem
println("Partitioning the source")

# partitioning the source % TODO make it a function
ff = sourcePartition(f, nx,ny,nz, npml,nzi,nLayer );

# for loop for solving each system this should be done in parallel
uArray = {solve(subArray[ii], ff[ii]) for ii=1:nLayer};

# obdatin all the boundary data here (just be carefull with the first and last components)
uBdyData = {extractBoundaryData(subArray[ii], uArray[ii]) for ii=1:nLayer  };

# vectorizing using the polarized construction
uBdyPol = vectorizePolarizedBdyData(uBdyData);

##################################################################################
#  Solving for the boundary data

# building the Permutation matriz P
P = generatePermutationMatrix(nx,ny,nLayer );

# getting the rhs in the correct form (we need a minus in fron of it)
uBdyPer = -uBdyPol;

# gmres is a bit buggy and it needs to define an object that support
# for the preconditioners. In this case we build the preconditioned
# system by hand and we use gmres to solve it.
function PinvMM(v)
    uDummy =  PrecondGaussSeidel(subArray, P*(applyMM(subArray,v)),2)
    return uDummy
end

u0 = PrecondGaussSeidel(subArray,-uBdyPol,2);
u = u0;

using IterativeSolvers

##############  GMRES #####################
# solving for the traces
data = gmres!(u,PinvMM,u0; tol=0.0001);

println("Number of iteration of GMRES : ", countnz( data[2].residuals[:]))

#########################################################################
# testing that we obtained the good solution

# we apply the polarized matrix to u to check for the error
MMu = applyMM(subArray, u);

println("Error for the polarized boundary integral system = ", norm(MMu - P'*uBdyPer));

# adding the polarized traces to obtain the traces
# u = u^{\uparrow} + u^{\downarrow}
uBdySol = u[1:end/2]+u[(1+end/2):end];

uGamma = -vectorizeBdyData(uBdyData);

Mu = applyM(subArray, uBdySol);

# checking that we recover the good solution
println("Error for the boundary integral system = ", norm(Mu - uGamma));

########### Reconstruction ###############
# TODO this needs to be encapsulated too

# using the devectorization routine to use a simple for afterwards
uBdySolArray = devectorizeBdyData(subArray, uBdySol);

uSolArray = {reconstructLocally(subArray[ii],ff[ii], uBdySolArray[1][ii],
                                uBdySolArray[2][ii], uBdySolArray[3][ii],
                                uBdySolArray[4][ii]) for ii = 1:nLayer}

#concatenation fo the solution
uSol = zeros(Complex{Float64},nx,ny,nz);
uSol[:,:,1:(nzi+npml)] = reshape(uSolArray[1],nx,ny,nzi+2*npml)[:,:,1:(nzi+npml)];

for ii = 1:nLayer-2
    uSol[:,:,npml+(1:nzi)+ii*nzi] =  reshape(uSolArray[ii+1],nx,ny,nzi+2*npml)[:,:,npml+(1:nzi)];
end

ii = nLayer-1;
uSol[:,:,npml+(1:(nzi+npml))+ii*nzi] =  reshape(uSolArray[ii+1],nx,ny,nzi+2*npml)[:,:,npml+(1:(nzi+npml))];

# # We can build the global operator and test the error
# print("Assembling the Hemholtz Matrix \n")
# H = HelmholtzMatrix(m,nx,ny,nz,npml,h,fac,order,omega);

# sum(abs(H*uSol[:] - f[:]))


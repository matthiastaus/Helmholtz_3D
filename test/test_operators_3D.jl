#test case for all the constructions
# loading the functions needed to build the system
include("../src/subdomain.jl");

# number of deegres of freedom per dimension
nx = 50;
ny = 50;
nz = 60;
npml = 10;

# number of layers
nLayer = 4;

# interior degrees of freedom
nzi = int((nz-2*npml)/nLayer);
# extended deegres of freedom
nzd = nzi+2*npml;


z = linspace(0,1,nz);
zInd = {npml+1+nzi*(ii-1) for ii = 1:nLayer}

# the squared slowness, inthis case constant velocity
m = ones(nx,ny,nz);

# extra arguments
npml  = 10;
h     = z[2]-z[1];
fac   = 10/h;
order = 2;
K     = nx/10;
omega = 2*pi*K;

# building teh Global solver for debugging purpouses
Global_solver = Subdomain(m,npml,[0 0 0],h,fac,order,omega);


# defining the full source (point source in this case)
n = nx*ny*nz;
f = zeros(nx,ny,nz);
f[30,30,15] = n;


# obtaining the global solution
uGlobal = solve(Global_solver,f);

uGlobal = reshape(uGlobal,nx,ny,nz);
uGlobalBdy = uGlobal[:,:, [20,21,30,31,40,41] ][:] ;


# bulding the matrix
print("Assembling the Hemholtz Matrix \n")
subArray = { Subdomain(m[:,:,(1:nzd)+nzi*(ii-1)],npml,[0 0 z[11+10*(ii-1)]],h,fac,order,omega) for ii=1:nLayer};



# partitioning the source % TODO make it a function
ff = {};
ftem = zeros(nx,ny,nzd);
ftem[:,:,1:nzd-npml] = f[:,:,1:nzd-npml];
push!(ff,ftem);

for ii = 1:nLayer-2
    ftem = zeros(nx,ny,nzd);
    ftem[:,:,npml+(1:nzi)] = f[:,:,(1:nzi)+ii*nzi+npml];
    push!(ff,ftem);
end
ftem = zeros(nx,ny,nzd);
ftem[:,:,npml+(1:nzi+npml)] = f[:,:,(npml+(1:nzi+npml))+(nLayer-1)*nzi];
push!(ff,ftem);

# for loop for solving each system this should be done in parallel
uArray = {solve(subArray[ii], ff[ii]) for ii=1:nLayer};

# obdatin all the boundary data here (just be carefull with the first and last components)
uBdyData = {extractBoundaryData(subArray[ii], uArray[ii]) for ii=1:nLayer  };

## just apply M matrix function
# u in the boundaries
uGamma = vectorizeBdyData(uBdyData);

Mu = applyM(subArray, uGlobalBdy);

# checking that we recover the good solution
norm(Mu + uGamma)


#picking a random vector to apply the operators
uGamma = rand(size(uGamma));

# veryginf the different operators
MMuL = P*(applyMM(subArray, vcat(uGamma,0*uGamma)));
Lu  = applyL(subArray, uGamma);
Du  = applyDdown(subArray, uGamma);

norm(Du - MMuL[1:end/2])
norm(Lu - MMuL[(1+end/2):end])

MMuU = P*(applyMM(subArray, vcat(0*uGamma,uGamma)));
Uu  = applyU(subArray, uGamma);
Dup = applyDup(subArray,uGamma);

norm(Uu  - MMuU[1:end/2])
norm(Dup - MMuU[(1+end/2):end])

norm(uGamma - applyDinvDown(subArray, applyDdown(subArray,uGamma)))

norm(uGamma - applyDinvUp(subArray, applyDup(subArray,uGamma)))

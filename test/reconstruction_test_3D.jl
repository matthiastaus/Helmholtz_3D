# Script to test that the reconstruction is exact up to
# machine precission

# loading the functions needed to build the system
include("../src/subdomain.jl");

# number of deegres of freedom per dimension
nx = 40;
ny = 40;
nz = 50;
npml = 10;

# number of layers
nLayer = 3;

# interior degrees of freedom
nzi = int((nz-2*npml)/nLayer);
# extended deegres of freedom
nzd = nzi+2*npml;


z = linspace(0,1,nz);
zInd = {npml+1+nzi*(ii-1) for ii = 1:nLayer}

# the squared slowness, inthis case constant velocity
m = ones(nx,ny,nz);

# extra arguments
h     = z[2]-z[1];
fac   = 10/h;
order = 2;
K     = nx/10;
omega = 2*pi*K;

# building the Global solver for debugging purpouses
Global_solver = Subdomain(m,npml,[0 0 z[npml+1]],h,fac,order,omega);

# defining the full source (point source in this case)
n = nx*ny*nz;
f = zeros(nx,ny,nz);
f[round(Integer, nx/2) ,round(Integer, ny/2),round(Integer, nz/2)] = n;

# obtaining the global solution
uGlobal = solve(Global_solver,f);
uGlobal = reshape(uGlobal,nx,ny,nz);

# obdatin all the boundary data here (just be carefull with the first and last components)
uBdyData = extractBoundaryData(Global_solver, uGlobal[:]) ;

uu = reconstructLocally(Global_solver,f, uBdyData[1], uBdyData[2], uBdyData[3], uBdyData[4]);
uu = reshape(uu,nx,ny,nz);

uPmlup   = uu[:,:,1:npml] ;
uPmldown = uu[:,:,(end-npml+1):end];

size(uPmldown)
size(uPmlup)

norm(uPmlup[:])
norm(uPmldown[:])

error = uGlobal[:,:,(1:(nz-2*npml) )+npml] - uu[:,:,(1:(nz-2*npml) )+npml] ;

size(error)

norm(error[:])


misfit   = uGlobal - uu;
uBdyDataN = extractBoundaryData(Global_solver, misfit[:]) ;

println("Everything should be super small (14 digits)")

println(norm(uBdyDataN[2][:]))
println(norm(uBdyDataN[3][:]))
println(norm(uBdyDataN[1][:] - uBdyData[1][:]))
println(norm(uBdyDataN[4][:] - uBdyData[4][:]))

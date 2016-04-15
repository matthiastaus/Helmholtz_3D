# Script to test the method of polarized traces to
# solve the 3D Helmholtz equation
# The wavespeed is a toy fault model 

# loading the functions needed to build the system
include("../src/subdomain.jl");
include("../src/preconditioners.jl")
using IterativeSolvers

# number of deegres of freedom per dimension
nx = 40;
ny = 56;
nz = 56;
npml = 8;

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
fac   = 30/(npml*h);  # absorbition coefficient for the PML
order = 2;            # order of the method, this should be changes
K     = nz/6;         # wave number
omega = 2*pi*K;       # frequency

# the squared slowness, we have planes plus a oblique fracture plus some Gaussian bumps
m = zeros(Float64,nx,ny,nz);


# we compute the speed and then the squared slownedd
c = zeros(Float64,nx,ny,nz);

function bump(x::Float64,y::Float64,z::Float64,center, amplitude::Float64, spread::Float64, orientation)
    return amplitude*exp( -( orientation[1]*(x-center[1]).^2 +
                             orientation[2]*(y-center[2]).^2 + 
                             orientation[3]*(z-center[3]).^2)/spread )  
end

function plane(x::Float64,y::Float64,z::Float64, point, vector)
  return  ((x-point[1])*vector[1] + (y - point[2])*vector[2] + (z - point[3])*vector[3])  >= 0
end

function layer(x::Float64,y::Float64,z::Float64,points, vectors, amplitude)
    return amplitude*plane(x,y,z, points[1,:], vectors[1,:])*
                     plane(x,y,z, points[2,:], vectors[2,:])* 
                     plane(x,y,z, points[3,:], vectors[3,:])
end

vvs = [ 0.0  0.0   1.0;
        0.0  0.0  -1.0;
        0.0  sqrt(2)   sqrt(2)];
vvn =  [ 0.0  0.0   1.0;
        0.0  0.0  -1.0;
        0.0  -sqrt(2)   -sqrt(2)];

vvtop = [ 0.0  0.0   1.0;
          0.0  0.0  -1.0;
          0.0  0.0   1.0];

p1 = [ 0 0 0.2 ;
       0 0 0.41 ;
       0.5 0.75 0.25];

p2 = [0 0 0.410001 ;
       0 0 0.61 ;
       0.5 0.5 0.5]; 

p3 = [0 0 0.610001 ;
       0 0 0.85
       0.5 0.25 0.75]; 
    
p4 = [0 0 0.850001 ;
       0 0 1.1
       0 0 0.850001]; 

p1s = [ 0 0 0.25 ;
       0 0 0.45 ;
       0.5 0.75 0.25];

p2s= [0 0 0.450001 ;
       0 0 0.70 ;
       0.5 0.5 0.5];   

p3s = [0 0 0.7000001 ;
       0 0 0.85
       0.5 0.25 0.75];           

# this is using a transverse approach 
for ii=1:nx,jj=1:ny,kk=1:nz 
    c[ii,kk,jj] = 1.2 + layer(ii*h,jj*h,kk*h,p1, vvs ,0.75)+
                         layer(ii*h,jj*h,kk*h,p1s, vvn ,0.75)+ 
                         layer(ii*h,jj*h,kk*h,p2, vvs ,1.25)+
                         layer(ii*h,jj*h,kk*h,p2s, vvn ,1.25) +
                         layer(ii*h,jj*h,kk*h,p3, vvs ,1.75)+
                         layer(ii*h,jj*h,kk*h,p3s,vvn ,1.75) +
                          layer(ii*h,jj*h,kk*h,p4,vvtop ,2.3) + 
                          bump(ii*h,jj*h,kk*h, [0.70  0.64  0.89], -0.01, 0.018, [0.39  2.45  0.78] ) + 
                          bump(ii*h,jj*h,kk*h, [0.03  0.70  0.95], 0.3 , 0.026, [2.82  2.38  1.80] ) + 
                          bump(ii*h,jj*h,kk*h, [0.27  0.75  0.54], -0.02, 0.014, [2.86  1.93  2.13] ) + 
                          bump(ii*h,jj*h,kk*h, [0.04  0.27  0.13], 0.04 , 0.013, [1.72  1.13  0.66] ) + 
                          bump(ii*h,jj*h,kk*h, [0.09  0.67  0.14], -0.03, 0.086, [0.17  2.43  0.35] )+ 
                          bump(ii*h,jj*h,kk*h, [0.82  0.65  0.25], 0.4 , 0.057, [0.70  1.59  0.89] )+ 
                          bump(ii*h,jj*h,kk*h, [0.69  0.16  0.84], .2 , 0.054, [1.05  1.05  0.95]  )+ 
                          bump(ii*h,jj*h,kk*h, [0.31  0.11  0.25], 0.06 , 0.014, [2.46  2.81  1.27]  ) + 
                          bump(ii*h,jj*h,kk*h, [0.95  0.49  0.81], -0.02, 0.085, [0.04  2.62  1.52] ) + 
                          bump(ii*h,jj*h,kk*h, [0.03  0.95  0.24], -0.01, 0.062, [0.12  1.65  0.25] ) + 
                          bump(ii*h,jj*h,kk*h, [0.43  0.34  0.92], 0.03 , 0.035, [0.50  1.86  0.78] ) + 
                          bump(ii*h,jj*h,kk*h, [0.38  0.58  0.35], -0.02, 0.051, [1.94  1.76  2.40] ) + 
                          bump(ii*h,jj*h,kk*h, [0.76  0.22  0.19], 0.04 , 0.040, [2.19  0.62  0.08] ) + 
                          bump(ii*h,jj*h,kk*h, [0.79  0.75  0.25], -0.03, 0.007, [1.94  0.90  2.78] )+ 
                          bump(ii*h,jj*h,kk*h, [0.18  0.25  0.61], 0.04 , 0.023, [1.35  1.41  2.19] )+ 
                          bump(ii*h,jj*h,kk*h, [0.48  0.50  0.47], 0.05 , 0.012, [1.64  0.69  1.46]  )+
                          bump(ii*h,jj*h,kk*h, [0.83  0.01  0.82], -0.01, 0.018, [0.88  2.53  1.73] ) + 
                          bump(ii*h,jj*h,kk*h, [0.58  0.33  0.53], 0.13 , 0.024, [2.23  0.58  0.71] ) + 
                          bump(ii*h,jj*h,kk*h, [0.54  0.16  0.99], -0.02, 0.041, [0.56  0.67  1.37] ) + 
                          bump(ii*h,jj*h,kk*h, [0.91  0.79  0.07], 0.04 , 0.004, [2.06  0.51  2.88] ) + 
                          bump(ii*h,jj*h,kk*h, [0.28  0.31  0.44], -0.03, 0.090, [0.55  0.68  1.64] )+ 
                          bump(ii*h,jj*h,kk*h, [0.75  0.52  0.10], 0.04 , 0.094, [1.10  1.30  1.56] )+ 
                          bump(ii*h,jj*h,kk*h, [0.75  0.16  0.96], 0.05 , 0.049, [1.87  0.93  0.69]  )+ 
                          bump(ii*h,jj*h,kk*h, [0.38  0.60  0.00], 0.06 , 0.048, [2.34  2.77  1.46]  ) + 
                          bump(ii*h,jj*h,kk*h, [0.56  0.26  0.77], -0.02, 0.033, [0.24  1.29  1.87] ) + 
                          bump(ii*h,jj*h,kk*h, [0.07  0.65  0.81], -0.01, 0.090, [2.78  0.55  2.03] ) + 
                          bump(ii*h,jj*h,kk*h, [0.05  0.68  0.86], 0.03 , 0.036, [2.32  2.71  1.18] ) + 
                          bump(ii*h,jj*h,kk*h, [0.53  0.74  0.08], -0.2, 0.011, [1.46  2.93  1.10] ) + 
                          bump(ii*h,jj*h,kk*h, [0.77  0.45  0.39], 0.04 , 0.078, [1.30  1.31  2.96] ) + 
                          bump(ii*h,jj*h,kk*h, [0.93  0.08  0.25], -0.03, 0.038, [1.34  0.33  0.11] )+ 
                          bump(ii*h,jj*h,kk*h, [0.12  0.22  0.80], 0.04 , 0.024, [0.91  0.77  2.65] )+ 
                          bump(ii*h,jj*h,kk*h, [0.09  0.71  0.29], 0.05 , 0.0047, [2.00  1.27  2.62]  )+ 
                          bump(ii*h,jj*h,kk*h, [0.26  0.50  0.43], 0.03 , 0.0015, [0.57  0.93  1.55] ) + 
                          bump(ii*h,jj*h,kk*h, [0.33  0.47  0.01], -0.02, 0.0034, [1.10  0.48  2.83] ) + 
                          bump(ii*h,jj*h,kk*h, [0.67  0.05  0.98], 0.4 , 0.0061, [1.38  0.53  1.91] ) + 
                          bump(ii*h,jj*h,kk*h, [0.13  0.68  0.16], -0.3, 0.0019, [2.94  1.26  2.87] )+ 
                          bump(ii*h,jj*h,kk*h, [0.72  0.04  0.10], 0.4 , 0.0074, [0.46  0.28  0.72] )+ 
                          bump(ii*h,jj*h,kk*h, [0.10  0.07  0.37], 0.02 , 0.0024, [2.56  1.79  2.02]  )+ 
                          bump(ii*h,jj*h,kk*h, [0.65  0.52  0.19], 0.06 , 0.0092, [1.93  1.41  0.86]  ) + 
                          bump(ii*h,jj*h,kk*h, [0.49  0.09  0.48], -0.2, 0.0027, [1.12  2.08  2.01] ) + 
                          bump(ii*h,jj*h,kk*h, [0.77  0.81  0.33], -0.1, 0.0077, [0.57  2.09  2.08] ) + 
                          bump(ii*h,jj*h,kk*h, [0.71  0.81  0.95], 0.3 , 0.0019, [1.28  1.91  0.20] ) + 
                          bump(ii*h,jj*h,kk*h, [0.90  0.72  0.92], -0.02, 0.0029, [1.44  0.10  0.76] ) + 
                          bump(ii*h,jj*h,kk*h, [0.89  0.14  0.05], 0.04 , 0.0009, [0.36  0.20  0.67] ) + 
                          bump(ii*h,jj*h,kk*h, [0.33  0.65  0.73], -0.03, 0.0058, [1.76  0.95  2.00] )+ 
                          bump(ii*h,jj*h,kk*h, [0.69  0.51  0.26], 0.04 , 0.0068, [0.67  1.59  2.53] )+ 
                          bump(ii*h,jj*h,kk*h, [0.19  0.97  0.42], 0.05 , 0.0055, [1.15  1.96  1.03]  )+
                          bump(ii*h,jj*h,kk*h, [0.03  0.64  0.54], -0.01, 0.0043, [1.74  1.22  2.34] ) + 
                          bump(ii*h,jj*h,kk*h, [0.74  0.80  0.94], 0.03 , 0.0064, [0.75  2.45  2.02] ) + 
                          bump(ii*h,jj*h,kk*h, [0.50  0.45  0.41], -0.02, 0.0065, [0.87  2.15  0.02] ) + 
                          bump(ii*h,jj*h,kk*h, [0.47  0.43  0.98], 0.04 , 0.0068, [1.85  2.90  1.80] ) + 
                          bump(ii*h,jj*h,kk*h, [0.90  0.82  0.30], -0.03, 0.0064, [0.79  1.59  1.16] )+ 
                          bump(ii*h,jj*h,kk*h, [0.60  0.08  0.70], 0.04 , 0.0095, [2.47  0.97  2.74] )+ 
                          bump(ii*h,jj*h,kk*h, [0.61  0.13  0.66], 0.05 , 0.0021, [2.94  0.31  0.00]  )+ 
                          bump(ii*h,jj*h,kk*h, [0.85  0.17  0.53], 0.06 , 0.0071, [2.19  1.83  1.38]  ) + 
                          bump(ii*h,jj*h,kk*h, [0.80  0.39  0.69], -0.02, 0.0024, [1.03  2.33  1.27] ) + 
                          bump(ii*h,jj*h,kk*h, [0.57  0.83  0.66], -0.1, 0.0012, [1.75  1.27  1.38] ) + 
                          bump(ii*h,jj*h,kk*h, [0.18  0.80  0.17], 0.03 , 0.0061, [0.32  0.27  2.31] ) + 
                          bump(ii*h,jj*h,kk*h, [0.23  0.06  0.12], -0.02, 0.0045, [2.71  0.79  0.96] ) + 
                          bump(ii*h,jj*h,kk*h, [0.88  0.39  0.99], 0.04 , 0.0046, [2.63  0.46  2.35] ) + 
                          bump(ii*h,jj*h,kk*h, [0.02  0.52  0.17], -0.03, 0.0066, [2.45  0.84  1.41] )+ 
                          bump(ii*h,jj*h,kk*h, [0.48  0.41  0.03], 0.4 , 0.0077, [0.78  1.32  0.10] )+ 
                          bump(ii*h,jj*h,kk*h, [0.16  0.65  0.56], 0.5 , 0.0035, [1.78  1.58  0.52]  )+
                          bump(ii*h,jj*h,kk*h, [0.97  0.62  0.88], 0.06 , 0.0066, [0.06  1.37  2.16]  ) +
                          bump(ii*h,jj*h,kk*h, [0.83  0.52  0.45],  0.1890, 0.018, [0.39  2.45  0.78] ) + 
                          bump(ii*h,jj*h,kk*h, [0.43  0.37  0.20],  0.1158, 0.026, [2.82  2.38  1.80] ) + 
                          bump(ii*h,jj*h,kk*h, [0.47  0.93  0.89],  0.1211, 0.014, [2.86  1.93  2.13] ) + 
                          bump(ii*h,jj*h,kk*h, [0.56  0.82  0.76],  0.0842, 0.013, [1.72  1.13  0.66] ) + 
                          bump(ii*h,jj*h,kk*h, [0.26  0.84  0.88],  0.1739, 0.086, [0.17  2.43  0.35] )+ 
                          bump(ii*h,jj*h,kk*h, [0.74  0.37  0.28],  0.0760, 0.057, [0.70  1.59  0.89] )+ 
                          bump(ii*h,jj*h,kk*h, [0.50  0.59  0.67],  0.1892, 0.054, [1.05  1.05  0.95]  )+ 
                          bump(ii*h,jj*h,kk*h, [0.64  0.87  0.66], -0.1921, 0.014, [2.46  2.81  1.27]  ) + 
                          bump(ii*h,jj*h,kk*h, [0.30  0.93  0.12],  0.2029, 0.085, [0.04  2.62  1.52] ) + 
                          bump(ii*h,jj*h,kk*h, [0.13  0.66  0.40], -0.02150, 0.062, [0.12  1.65  0.25] ) + 
                          bump(ii*h,jj*h,kk*h, [0.47  0.20  0.27],  0.0505, 0.035, [0.50  1.86  0.78] ) + 
                          bump(ii*h,jj*h,kk*h, [0.36  0.65  0.71],  0.2263, 0.051, [1.94  1.76  2.40] ) + 
                          bump(ii*h,jj*h,kk*h, [0.78  0.07  0.28],  0.0788, 0.040, [2.19  0.62  0.08] ) + 
                          bump(ii*h,jj*h,kk*h, [0.78  0.40  0.89], -0.01180, 0.007, [1.94  0.90  2.78] )+ 
                          bump(ii*h,jj*h,kk*h, [0.66  0.66  0.82], -0.1473, 0.023, [1.35  1.41  2.19] )+ 
                          bump(ii*h,jj*h,kk*h, [0.13  0.93  0.39], -0.0225, 0.012, [1.64  0.69  1.46]  )+
                          bump(ii*h,jj*h,kk*h, [0.02  0.81  0.49], -0.0931, 0.018, [0.88  2.53  1.73] ) + 
                          bump(ii*h,jj*h,kk*h, [0.55  0.48  0.69], -0.1968, 0.024, [2.23  0.58  0.71] ) + 
                          bump(ii*h,jj*h,kk*h, [0.30  0.75  0.83],  0.2226, 0.041, [0.56  0.67  1.37] ) + 
                          bump(ii*h,jj*h,kk*h, [0.93  0.41  0.60],  0.0982, 0.004, [2.06  0.51  2.88] ) + 
                          bump(ii*h,jj*h,kk*h, [0.98  0.97  0.57],  0.2269, 0.090, [0.55  0.68  1.64] )+ 
                          bump(ii*h,jj*h,kk*h, [0.28  0.98  0.32],  0.1523, 0.094, [1.10  1.30  1.56] )+ 
                          bump(ii*h,jj*h,kk*h, [0.80  0.86  0.45], -0.1101, 0.049, [1.87  0.93  0.69]  )+ 
                          bump(ii*h,jj*h,kk*h, [0.89  0.38  0.71], -0.1109, 0.048, [2.34  2.77  1.46]  ) + 
                          bump(ii*h,jj*h,kk*h, [0.59  0.45  0.88], -0.1889, 0.033, [0.24  1.29  1.87] ) + 
                          bump(ii*h,jj*h,kk*h, [0.88  0.24  0.72], -0.0412, 0.090, [2.78  0.55  2.03] ) + 
                          bump(ii*h,jj*h,kk*h, [0.94  0.78  0.01],  0.2147, 0.036, [2.32  2.71  1.18] ) + 
                          bump(ii*h,jj*h,kk*h, [0.54  0.88  0.67], -0.2114, 0.011, [1.46  2.93  1.10] ) + 
                          bump(ii*h,jj*h,kk*h, [0.72  0.91  0.43], -0.1502, 0.078, [1.30  1.31  2.96] ) + 
                          bump(ii*h,jj*h,kk*h, [0.57  0.55  0.43],  0.1070, 0.038, [1.34  0.33  0.11] )+ 
                          bump(ii*h,jj*h,kk*h, [0.02  0.59  0.11], -0.0218, 0.024, [0.91  0.77  2.65] )+ 
                          bump(ii*h,jj*h,kk*h, [0.44  0.14  0.81], -0.2424, 0.0047, [2.00  1.27  2.62]  )+ 
                          bump(ii*h,jj*h,kk*h, [0.64  0.89  0.32], -0.1078, 0.0015, [0.57  0.93  1.55] )  ;
end

m = (1./c).^2;
c = 0;

gc();


println("Solving the Helmholtz equation for nx = ",nx,", ny = ",ny, ", nz = ",nz,
        ", npml = ", npml, ", nLayer = ", nLayer, ", internal deegres of freedom = ", nzi,
        ", absorbtion factor = ", fac , " at frequency = ", omega);

# defining the full source (point source in this case)
n = nx*ny*nz;
f = zeros(nx,ny,nz);
# due to a bug the source needs to be in the top layer
f[8,8,18] = n;


# bulding the domain decomposition data structure
println("Building the subdomains and factorizing the problems locally \n")
subArray = [Subdomain(m[:,:,(1:nzd)+nzi*(ii-1)],npml,[0 0 z[1+npml+nzi*(ii-1)]],
			 h,fac,order,omega) for ii=1:nLayer];


####################################################################################
# Solving the local problem
println("Partitioning the source")

# partitioning the source % TODO make it a function
ff = sourcePartition(f, nx,ny,nz, npml,nzi,nLayer );

# for loop for solving each system this should be done in parallel
uArray = [solve(subArray[ii], ff[ii]) for ii=1:nLayer];

# obdatin all the boundary data here (just be carefull with the first and last components)
uBdyData = [extractBoundaryData(subArray[ii], uArray[ii]) for ii=1:nLayer  ];

# vectorizing using the polarized construction
# there is a bug inside this function!!!! 
uBdyPol = vectorizePolarizedBdyDataRHS(uBdyData);

##################################################################################
#  Solving for the boundary data

# building the Permutation matriz P
P = generatePermutationMatrix(nx,ny,nLayer );

# getting the rhs in the correct form (we need a minus in fron of it)
uBdyPer = -uBdyPol;

# allocating the preconditioner
Precond = PolarizedTracesPreconditioner(subArray, P)

##############  GMRES #####################
# # solving for the traces

u = 0*uBdyPol;
@time data = gmres!(u,x->applyMMOpt(subArray,x), uBdyPer, Precond; tol=0.00001);


println("Number of iteration of GMRES : ", countnz( data[2].residuals[:]))

#########################################################################
# testing the solution

# we apply the polarized matrix to u to check for the error
MMu = applyMMOpt(subArray, u);

println("Error for the polarized boundary integral system = ", norm(MMu - uBdyPer));

# adding the polarized traces to obtain the traces
# u = u^{\uparrow} + u^{\downarrow}
uBdySol = u[1:round(Integer,end/2)]+u[(1+round(Integer,end/2)):end];

uGamma = -vectorizeBdyData(uBdyData);

Mu = applyM(subArray, uBdySol);

# checking that we recover the good solution
println("Error for the boundary integral system = ", norm(Mu - uGamma)/norm(uGamma));

########### Reconstruction ###############
# TODO this needs to be encapsulated too

# using the devectorization routine to use a simple for afterwards
uBdySolArray = devectorizeBdyData(subArray, uBdySol);

uSolArray = [reconstructLocally(subArray[ii],ff[ii], uBdySolArray[1][ii],
                                uBdySolArray[2][ii], uBdySolArray[3][ii],
                                uBdySolArray[4][ii]) for ii = 1:nLayer];

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
#H = HelmholtzMatrix(m,nx,ny,nz,npml,h,fac,order,omega);

#norm(H*uSol[:] - f[:])/norm(f[:])


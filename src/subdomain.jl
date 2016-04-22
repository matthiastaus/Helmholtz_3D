# Subdomain type
# the main idea behind is to abstract all the information needed to perform the 
# method of polarized traces. 

include("HelmholtzMatrix.jl");

type Subdomain
    H::SparseMatrixCSC{Complex{Float64},Int64} # sparse matrix
    # local meshes
    h::Float64
    x     #mesh in x
    y
    z
    #local delimiters
    xLim    # (x0 x1 xN xNp)
    yLim    # (y0 y1 yN yNp)
    zLim
    # indeces for the local boundary elements
    xLimInd::Array{Int64,2}
    yLimInd::Array{Int64,2}
    zLimInd::Array{Int64,2}
    Hinv
    size
    function Subdomain(m,npml::Int,bdy,h::Float64,fac::Float64,order::Int,omega::Float64)
        # m = matrix(nx,ny,nz)
        # extracting the size of the 3D domain
        (nx,ny,nz) = size(m);
        # building the differential operator
        H = HelmholtzMatrix(m,nx,ny,nz,npml,h,fac,order,omega);
        # building the grids in each dimension
        x = linspace(bdy[1]-h*(npml), bdy[1]+h*(nx-npml-1), nx );
        y = linspace(bdy[2]-h*(npml), bdy[2]+h*(ny-npml-1), ny );
        z = linspace(bdy[3]-h*(npml), bdy[3]+h*(nz-npml-1), nz );
        # building the boundary data
        xLim = [ x[npml] x[npml+1] x[nx-npml] x[nx-npml+1]];
        yLim = [ y[npml] y[npml+1] y[ny-npml] y[ny-npml+1]];
        zLim = [ z[npml] z[npml+1] z[nz-npml] z[nz-npml+1]];

        zLimInd= [ (collect(1:nx*ny) + (npml-1)*nx*ny    ).'  ;
                   (collect(1:nx*ny) + (npml)*nx*ny      ).' ;
                   (collect(1:nx*ny) + (nz-npml-1)*nx*ny ).' ;
                   (collect(1:nx*ny) + (nz-npml)*nx*ny   ).' ; ];

        new(H,h,x,y,z,xLim,yLim,zLim,[7 3],[8 2],zLimInd, lufact(H), [nx*ny*nz,nx,ny,nz]) # don't know if it's the best answer
    end
end

function solve(subdomain::Subdomain, f) # put a flag to use MUMPS instead of UMFPACK
    # u = solve(subdomain::Subdomain, f)
    # function that solves the system Hu=f in the subdomain
    # check size
    if (size(f[:])[1] == subdomain.size[1])
        u = subdomain.Hinv\f[:];
        return u
    else
        print("The dimensions do not match \n");
        return 0
    end
end

function extractBoundaryData(subdomain::Subdomain, u)
    # Function to extract the boundary data from a solution u
    # input   subdomain: subdomain associated to the solution
    #         u        : solution
    # output  (u0, u1, uN, uNp) : tuple of the solution at different depth
    # check size
    if (size(u[:])[1] == subdomain.size[1])
        u0  = u[subdomain.zLimInd[1,:]];
        u1  = u[subdomain.zLimInd[2,:]];
        uN  = u[subdomain.zLimInd[3,:]];
        uNp = u[subdomain.zLimInd[4,:]];
        return (u0,u1,uN,uNp)
    else
        print("Dimension mismatch \n");
        return 0
    end
end

function applyBlockOperator(subdomain::Subdomain,v0,v1,vN,vNp)
    # function to apply the local matricial operator to the interface data
    # and we sample it at the interface
    # allocating the source
    f = zeros(Complex{Float64},subdomain.size[2],subdomain.size[3],subdomain.size[4]);
    # filling the source with the correct single and double layer potentials
    f[subdomain.zLimInd[2,:]] = v0;
    f[subdomain.zLimInd[1,:]] = -v1;
    f[subdomain.zLimInd[4,:]] = -vN;
    f[subdomain.zLimInd[3,:]] = vNp;
    f = f*(1/subdomain.h)^2;
    u = solve(subdomain, f[:]);

    u0  = u[subdomain.zLimInd[1,:]];
    u1  = u[subdomain.zLimInd[2,:]];
    uN  = u[subdomain.zLimInd[3,:]];
    uNp = u[subdomain.zLimInd[4,:]];
    return (u0,u1,uN,uNp)

end


function reconstructLocally(subdomain::Subdomain,f, u0,u1,uN,uNp)
    # function to apply the local matricial operator to the interface data
    # and we sample it at the interface
    # allocating the source for the single and double layer potential
    g = zeros(Complex{Float64}, subdomain.size[2],subdomain.size[3],subdomain.size[4])[:];
    print("source allocated \n")
    # filling the source with the correct single and double layer potentials
    g[subdomain.zLimInd[2,:]] = u0;
    g[subdomain.zLimInd[1,:]] = -u1;
    g[subdomain.zLimInd[4,:]] = -uN;
    g[subdomain.zLimInd[3,:]] = uNp;
    print("single and double layer potential modified \n")
    g = g*(1/subdomain.h)^2;
    # adding both contributions, internal and external
    f = f[:] + g[:];
    u = solve(subdomain, f[:]);

    return u

end

function vectorizeBdyData(uBdyData)
    # function to take the output of extract Boundary data and put it in vectorized form
    nLayer = size(uBdyData)[1]
    # nn     = size(uBdyData[1][1])[1] # depends on the version
    nn     = length(uBdyData[1][1])

    uBdy = zeros(Complex{Float64},2*(nLayer-1)*nn);

    nInd = 1:nn;

    uBdy[nInd] = uBdyData[1][3];
    for ii = 1:nLayer-2
        uBdy[nInd+(2*ii-1)*nn] = uBdyData[ii+1][2];
        uBdy[nInd+2*ii*nn] = uBdyData[ii+1][3];
    end
    uBdy[nInd+(2*nLayer-3)*nn] = uBdyData[nLayer][2];

    return uBdy
end

function vectorizePolarizedBdyData(uBdyData)
    # function to take the output of extract Boundary data and put it in vectorized form
    nLayer = size(uBdyData)[1]
    nSurf  = length(uBdyData[1][1])

    uBdy = zeros(Complex{Float64},4*(nLayer-1)*nSurf);

    nInd = 1:nSurf;

    uBdy[nInd]       = uBdyData[1][3];
    uBdy[nInd+nSurf] = uBdyData[1][4];

    for ii = 1:nLayer-2
        uBdy[nInd+(4*ii-2)*nSurf] = uBdyData[ii+1][1];
        uBdy[nInd+(4*ii-1)*nSurf] = uBdyData[ii+1][2];
        uBdy[nInd+(4*ii  )*nSurf] = uBdyData[ii+1][3];
        uBdy[nInd+(4*ii+1)*nSurf] = uBdyData[ii+1][4];
    end
    ii = nLayer-1
    uBdy[nInd+(4*ii-2)*nSurf] = uBdyData[ii+1][1];
    uBdy[nInd+(4*ii-1)*nSurf] = uBdyData[ii+1][2];

    return uBdy
end

function vectorizePolarizedBdyDataRHS(uBdyData)
    # function to take the output of extract Boundary data and put it in vectorized form
    nLayer = size(uBdyData)[1]
    nSurf  = length(uBdyData[1][1])

    uBdy1 = zeros(Complex{Float64},2*(nLayer-1)*nSurf);
    uBdy0 = zeros(Complex{Float64},2*(nLayer-1)*nSurf);

    nInd = 1:nSurf;


    for ii = 1:nLayer-1
        uBdy1[nInd+(2*ii-2)*nSurf] = uBdyData[ii][3];
        uBdy1[nInd+(2*ii-1)*nSurf] = uBdyData[ii+1][2];
    end

    for ii = 1:nLayer-1
        uBdy0[nInd+(2*ii-2)*nSurf] = uBdyData[ii ][4];
        uBdy0[nInd+(2*ii-1)*nSurf] = uBdyData[ii+1][1];
    end

    return vcat(uBdy1,uBdy0)
end





function applyM(subArray, uGamma)
    # function to apply M to uGamma

    # decomposing the uGamma to be suitable for a condensed application
    (v0,v1,vN,vNp ) = devectorizeBdyData(subArray, uGamma)

    # applying
    Au = [ applyBlockOperator(subArray[ii],v0[ii],v1[ii],vN[ii],vNp[ii]) for ii = 1:nLayer ];

    # it need to be a vector
    Au = vectorizeBdyData(Au);

    Mu = Au - uGamma;

    return Mu[:]

end

# to be done!
function applyMup(subArray, uGamma)
    # function to apply M to uGamma
    # input subArray : an array of subdomains
    #       uGamma   : data on the boundaries in vector form

    # convert uGamma in a suitable vector to be applied
    (u0,u1,uN,uNp) = devectorizeBdyData(subArray, uGamma);

    # obtaining the number of layers
    nLayer = size(subArray)[1];
    nSurf = subArray[1].size[2]*subArray[1].size[3];
    nInd = 1:nSurf;

    # applying
    v1 =  [applyBlockOperator(subArray[ii],0*u0[ii],0*u1[ii],uN[ii],uNp[ii]) for ii = 1:nLayer] ;
    vN =  [applyBlockOperator(subArray[ii],  u0[ii],  u1[ii],uN[ii],uNp[ii]) for ii = 1:nLayer] ;

    Mu = zeros(Complex{Float64},2*(nLayer-1)*nSurf);

    Mu[nInd] = - uN[1] + vec(vN[1][3]);

    for ii=1:nLayer-2
        Mu[nInd + (2*ii-1)*nSurf] = - u1[ii+1] + vec(v1[ii+1][2]);
        Mu[nInd + (2*ii  )*nSurf] = - uN[ii+1] + vec(vN[ii+1][3]);
    end
    ii = nLayer-1;
    Mu[nInd + (2*ii-1)*nSurf] = - u1[ii+1];

    return Mu[:]

end

function applyMdown(subArray, uGamma)
    # function to apply M to uGamma
    # input subArray : an array of subdomains
    #       uGamma   : data on the boundaries in vector form

    # convert uGamma in a suitable vector to be applied
    (u0,u1,uN,uNp) = devectorizeBdyData(subArray, uGamma);

    # obtaining the number of layers
    nLayer = size(subArray)[1];
    nSurf = subArray[1].size[2]*subArray[1].size[3];
    nInd = 1:nSurf;

    # applying the local solves (for loop)
    v1 =  [applyBlockOperator(subArray[ii],u0[ii],u1[ii],  uN[ii],  uNp[ii]) for ii = 1:nLayer] ;
    vN =  [applyBlockOperator(subArray[ii],u0[ii],u1[ii],0*uN[ii],0*uNp[ii]) for ii = 1:nLayer] ;

    Mu = zeros(Complex{Float64},2*(nLayer-1)*nSurf);

    Mu[nInd] =  - uN[1];

    for ii=1:nLayer-2
        Mu[nInd + (2*ii-1)*nSurf] = - u1[ii+1] + vec(v1[ii+1][2]);
        Mu[nInd + (2*ii  )*nSurf] = - uN[ii+1] + vec(vN[ii+1][3]);
    end
    ii = nLayer-1;
    Mu[nInd + (2*ii-1)*nSurf] = - u1[ii+1] + vec(v1[ii+1][2]);

    return Mu[:]
end

function applyM0up(subArray, uGamma)
    # function to apply M to uGamma
    # input subArray : an array of subdomains
    #       uGamma   : data on the boundaries in vector form

    # convert uGamma in a suitable vector to be applied
    (u0,u1,uN,uNp) = devectorizeBdyData(subArray, uGamma);

    # obtaining the number of layers
    nLayer = size(subArray)[1];
    nSurf = subArray[1].size[2]*subArray[1].size[3];
    nInd = 1:nSurf;

    # applying
    # remember  v[ii][1] = v0[ii], v[ii][2] = v1[ii],
    #           v[ii][3] = vN[ii], v[ii][4] = vNp[ii]
    v1 =  [applyBlockOperator(subArray[ii],0*u0[ii],0*u1[ii],uN[ii],uNp[ii]) for ii = 1:nLayer] ;
    vN =  [applyBlockOperator(subArray[ii],  u0[ii],  u1[ii],uN[ii],uNp[ii]) for ii = 1:nLayer] ;

    Mu = zeros(Complex{Float64},2*(nLayer-1)*nSurf);

    Mu[nInd] =  vec(vN[1][4]);

    for ii=1:nLayer-2
        Mu[nInd + (2*ii-1)*nSurf] = - u0[ii+1] + vec(v1[ii+1][1]);
        Mu[nInd + (2*ii  )*nSurf] =            + vec(vN[ii+1][4]);
    end
    ii = nLayer-1;
    Mu[nInd + (2*ii-1)*nSurf] = -  u0[ii+1] + vec(v1[ii+1][1]);

    return Mu[:]

end

function applyM0down(subArray, uGamma)
    # function to apply M to uGamma
    # input subArray : an array of subdomains
    #       uGamma   : data on the boundaries in vector form

    # convert uGamma in a suitable vector to be applied
    (u0,u1,uN,uNp) = devectorizeBdyData(subArray, uGamma);

    # obtaining the number of layers
    nLayer = size(subArray)[1];
    nSurf = subArray[1].size[2]*subArray[1].size[3];
    nInd = 1:nSurf;

    # applying the local solves (for loop)
    v1 =  [applyBlockOperator(subArray[ii],u0[ii],u1[ii],  uN[ii],  uNp[ii]) for ii = 1:nLayer] ;
    vN =  [applyBlockOperator(subArray[ii],u0[ii],u1[ii],0*uN[ii],0*uNp[ii]) for ii = 1:nLayer] ;

    Mu = zeros(Complex{Float64},2*(nLayer-1)*nSurf);

    Mu[nInd] =  - uNp[1];

    for ii=1:nLayer-2
        Mu[nInd + (2*ii-1)*nSurf] =             + vec(v1[ii+1][1]);
        Mu[nInd + (2*ii  )*nSurf] = - uNp[ii+1] + vec(vN[ii+1][4]);
    end
    ii = nLayer-1;
    Mu[nInd + (2*ii-1)*nSurf] =   vec(v1[ii+1][1]);

    return Mu[:]
end

function applyMM(subArray, uGammaPol)
    # MMu = applyMM(subArray, uGammaPol)
    # function to apply the the polarized integral operator
    # by applying every single block
    println("Applying the polarized matrix")
    
    uDown = uGammaPol[1:round(Int64,end/2)];
    uUp = uGammaPol[(round(Int64,end/2)+1):end];

    MMu = vcat(applyMdown( subArray, uDown) + applyMup( subArray, uUp),
               applyM0down(subArray, uDown) + applyM0up(subArray, uUp));
    return MMu;

end



function reconstruction(subDomains, source, u0, u1, un, unp)
    #TODO add description 
    nSubs = length(subDomains);

    localSizes = zeros(Int64,nSubs)
    n = subDomains[1].model.size[2]*subDomains[1].model.size[3]
    # building the local rhs
    rhsLocal = [ zeros(Complex128,subDomains[ii].model.size[1]) for ii = 1:nSubs ]

    # copying the wave-fields
    for ii = 1:nSubs
    
        rhsLocal[ii][subDomains[ii].indVolIntLocal] = source[subDomains[ii].indVolInt]
        localSizes[ii] = length(subDomains[ii].indVolIntLocal)

    end

    # obtaining the limit of each subdomain within the global approximated solution
    localLim = [0; cumsum(localSizes)];

    uPrecond = zeros(Complex128, length(source))
    index = 1:n
    for ii = 1:nSubs
        

        ind_0  = subDomains[ii].ind_0
        ind_1  = subDomains[ii].ind_1
        ind_n  = subDomains[ii].ind_n
        ind_np = subDomains[ii].ind_np

        # making a copy of the parititioned source
        rhsLocaltemp = copy(rhsLocal[ii]);

        # adding the source at the boundaries
        if ii!= 1
            # we need to be carefull at the edges
            rhsLocaltemp[subDomains[ii].ind_1]  += -subDomains[ii].model.H[ind_1,ind_0]*u0[(ii-1)*n + index]
            rhsLocaltemp[subDomains[ii].ind_0]  +=  subDomains[ii].model.H[ind_0,ind_1]*u1[(ii-1)*n + index]

        end
        if ii!= nSubs
            rhsLocaltemp[subDomains[ii].ind_np] +=  subDomains[ii].model.H[ind_np,ind_n]*un[(ii-1)*n + index]
            rhsLocaltemp[subDomains[ii].ind_n]  += -subDomains[ii].model.H[ind_n,ind_np]*unp[(ii-1)*n + index]
        end


        uLocal = solve(subDomains[ii],rhsLocaltemp)

        uPrecond[localLim[ii]+1:localLim[ii+1]] = uLocal[subDomains[ii].indVolIntLocal]
    end
    return  uPrecond
end



#############################
#   Optimized functions     #
#                           #
#############################

function applyMMOpt(subDomains, uGamma)
    # function to apply M to uGamma in an optimized fashion
    # we can further optimize this function by feeding multiple right hand sides
    # however, built-in UMFPACK doesn not allow for multiple RHS then we do not 
    # perform that operatio
    # input subDomains : an array of subdomains
    #       uGamma   : data on the boundaries in vector form

    uDown = uGamma[1:round(Integer, end/2)]
    uUp   = uGamma[1+round(Integer, end/2):end]
    # convert uGamma in a suitable vector to be applied
    (u0Down,u1Down,uNDown,uNpDown) = devectorizeBdyData(subDomains, uDown);
    (u0Up  ,u1Up  ,uNUp  ,uNpUp)   = devectorizeBdyData(subDomains, uUp);

    # obtaining the number of layers
    nLayer = size(subDomains)[1];
    nSurf = subArray[1].size[2]*subArray[1].size[3];
    nInd = 1:nSurf;

    #TODO modify this functio in order to solve the three rhs in one shot
    # applying this has to be done in parallel applying RemoteRefs
    v1 = [ applyBlockOperator(subDomains[ii],u0Down[ii]         ,u1Down[ii]         ,
                              uNUp[ii]+uNDown[ii],uNpUp[ii]+uNpDown[ii]) for ii = 1:nLayer ];
    vN = [ applyBlockOperator(subDomains[ii],u0Down[ii]+u0Up[ii],u1Down[ii]+u1Up[ii],
                              uNUp[ii]           ,uNpUp[ii]            ) for ii = 1:nLayer ];

    Mu1 = zeros(Complex{Float64},2*(nLayer-1)*nSurf);

    Mu1[nInd] = - uNUp[1] - uNDown[1] + vec(vN[1][3]);

    for ii=1:nLayer-2
        Mu1[nInd + (2*ii-1)*nSurf] = - u1Up[ii+1] - u1Down[ii+1] + vec(v1[ii+1][2]);
        Mu1[nInd + (2*ii  )*nSurf] = - uNUp[ii+1] - uNDown[ii+1] + vec(vN[ii+1][3]);
    end
    ii = nLayer-1;
    Mu1[nInd + (2*ii-1)*nSurf] = - u1Up[ii+1] - u1Down[ii+1] + vec(v1[ii+1][2]) ;


    v1 = [ applyBlockOperator(subDomains[ii],u0Down[ii],u1Down[ii],uNUp[ii]+uNDown[ii],uNpUp[ii]+uNpDown[ii]) for ii = 1:nLayer ];
    vN = [ applyBlockOperator(subDomains[ii],u0Up[ii]+u0Down[ii],u1Down[ii]+u1Up[ii],uNUp[ii],uNpUp[ii]) for ii = 1:nLayer ];

    Mu = zeros(Complex{Float64},2*(nLayer-1)*nSurf);


    Mu[nInd] =  vec(vN[1][4])- uNpDown[1] ;

    for ii=1:nLayer-2
        Mu[nInd + (2*ii-1)*nSurf] = - u0Up[ii+1] + vec(v1[ii+1][1]);
        Mu[nInd + (2*ii  )*nSurf] = - uNpDown[ii+1] + vec(vN[ii+1][4]);
    end
    ii = nLayer-1;
    Mu[nInd + (2*ii-1)*nSurf] = -  u0Up[ii+1] + vec(v1[ii+1][1])

    return vcat(Mu1,Mu)

end



function applyL(subArray, uGamma)
    # function to apply M to uGamma
    # input subArray : an array of subdomains
    #       uGamma   : data on the boundaries in vector form

    # convert uGamma in a suitable vector to be applied
    (u0,u1,uN,uNp) = devectorizeBdyData(subArray, uGamma);

    # obtaining the number of layers
    nLayer = size(subArray)[1];
    nSurf = subArray[1].size[2]*subArray[1].size[3];
    nInd = 1:nSurf;

    # applying the local solves (for loop)
    v1 =  [applyBlockOperator(subArray[ii],u0[ii],u1[ii],  uN[ii],  uNp[ii]) for ii = 1:nLayer] ;

    Lu = zeros(Complex{Float64},2*(nLayer-1)*nSurf);

    for ii=1:nLayer-2
        Lu[nInd + (2*ii-2)*nSurf]  = vec(v1[ii+1][1])           ;
        Lu[nInd + (2*ii-1 )*nSurf] = vec(v1[ii+1][2]) - u1[ii+1];
    end
    ii = nLayer-1;

    Lu[nInd + (2*ii-2)*nSurf]  = vec(v1[ii+1][1])            ;
    Lu[nInd + (2*ii-1 )*nSurf] = vec(v1[ii+1][2]) - u1[ii+1] ;

    return Lu
end

function applyDdown(subArray, uGamma)
    # obtaining the number of layers
    nLayer = size(subArray)[1];
    nSurf = subArray[1].size[2]*subArray[1].size[3];
    nInd = 1:nSurf;

    dummyzero = zeros(Complex{Float64},nSurf);
    Du = zeros(Complex{Float64},2*(nLayer-1)*nSurf);

    Du[nInd]        = -uGamma[nInd];
    Du[nInd+nSurf]  = -uGamma[nInd+nSurf];

    for ii=1:nLayer-2
        uNm  = uGamma[nInd+ (2*ii-2)*nSurf];
        uNpm = uGamma[nInd+ (2*ii-1)*nSurf];
        uN   = uGamma[nInd+ (2*ii  )*nSurf];
        uNp  = uGamma[nInd+ (2*ii+1)*nSurf];

        (v0, v1, vN, vNp) = applyBlockOperator(subArray[ii+1],uNm,uNpm, dummyzero,dummyzero);

        Du[nInd+ 2*ii*nSurf]      = vec(vN)  - vec(uN);
        Du[nInd+ (2*ii+1)*nSurf]  = vec(vNp) - vec(uNp);

    end

    return Du
end

function applyDinvDown(subArray, uGamma)
    # obtaining the number of layers
    nLayer = size(subArray)[1];
    nSurf = subArray[1].size[2]*subArray[1].size[3];
    nInd = 1:nSurf;

    dummyzero = zeros(Complex{Float64},nSurf);
    Dinvu = zeros(Complex{Float64},2*(nLayer-1)*nSurf);

    Dinvu[nInd]        = -uGamma[nInd];
    Dinvu[nInd+nSurf]  = -uGamma[nInd+nSurf];
    vN   = Dinvu[nInd] ;
    vNp  = Dinvu[nInd+nSurf];

    for ii=1:nLayer-2
        uN = uGamma[nInd+ 2*ii*nSurf];
        uNp= uGamma[nInd+ (2*ii+1)*nSurf];

        (v0, v1, vN, vNp) = applyBlockOperator(subArray[ii+1],vN,vNp, dummyzero,dummyzero);

        Dinvu[nInd+ 2*ii*nSurf]      = vec(vN)  - vec(uN);
        Dinvu[nInd+ (2*ii+1)*nSurf]  = vec(vNp) - vec(uNp);

        vN   = Dinvu[nInd+ 2*ii*nSurf] ;
        vNp  = Dinvu[nInd+ (2*ii+1)*nSurf];
    end

    return Dinvu
end


function applyDup(subArray, uGamma)
    # obtaining the number of layers
    nLayer  = size(subArray)[1];
    nSurf   = subArray[1].size[2]*subArray[1].size[3];
    nInd    = 1:nSurf;

    dummyzero = zeros(Complex{Float64},nSurf);
    Dup     = zeros(Complex{Float64},2*(nLayer-1)*nSurf);

    jj = nLayer-1
    Dup[nInd+ (2*jj-2)*nSurf]  = -uGamma[nInd+ (2*jj-2)*nSurf];
    Dup[nInd+ (2*jj-1)*nSurf]  = -uGamma[nInd+ (2*jj-1)*nSurf];

    for ii = nLayer-2:-1:1
        u0m = uGamma[nInd+ (2*ii  )*nSurf];
        u1m = uGamma[nInd+ (2*ii+1)*nSurf];
        u0  = uGamma[nInd+ (2*ii-2)*nSurf];
        u1  = uGamma[nInd+ (2*ii-1)*nSurf];

        (v0, v1, vN, vNp) = applyBlockOperator(subArray[ii+1],dummyzero,dummyzero,u0m,u1m);

        Dup[nInd+ (2*ii-2)*nSurf]  = vec(v0) - vec(u0);
        Dup[nInd+ (2*ii-1)*nSurf]  = vec(v1) - vec(u1);

    end

    return Dup
end

function applyDinvUp(subArray, uGamma)
    # obtaining the number of layers
    nLayer  = size(subArray)[1];
    nSurf   = subArray[1].size[2]*subArray[1].size[3];
    nInd    = 1:nSurf;

    dummyzero = zeros(Complex{Float64},nSurf);
    Dinvu     = zeros(Complex{Float64},2*(nLayer-1)*nSurf);

    jj = nLayer-1
    Dinvu[nInd+ (2*jj-2)*nSurf]  = -uGamma[nInd+ (2*jj-2)*nSurf];
    Dinvu[nInd+ (2*jj-1)*nSurf]  = -uGamma[nInd+ (2*jj-1)*nSurf];
    v0  = Dinvu[nInd+ (2*jj-2)*nSurf] ;
    v1  = Dinvu[nInd+ (2*jj-1)*nSurf];

    for ii = nLayer-2:-1:1
        u0 = uGamma[nInd+ (2*ii-2)*nSurf];
        u1 = uGamma[nInd+ (2*ii-1)*nSurf];

        (v0, v1, vN, vNp) = applyBlockOperator(subArray[ii+1],dummyzero,dummyzero,v0,v1);

        Dinvu[nInd+ (2*ii-2)*nSurf]  = vec(v0) - vec(u0);
        Dinvu[nInd+ (2*ii-1)*nSurf]  = vec(v1) - vec(u1);

        v0   = Dinvu[nInd+ (2*ii-2)*nSurf] ;
        v1  = Dinvu[nInd+ (2*ii-1)*nSurf];
    end

    return Dinvu
end


function applyU(subArray, uGamma)
    # function to apply M to uGamma
    # input subArray : an array of subdomains
    #       uGamma   : data on the boundaries in vector form

    # convert uGamma in a suitable vector to be applied
    (u0,u1,uN,uNp) = devectorizeBdyData(subArray, uGamma);

    # obtaining the number of layers
    nLayer = size(subArray)[1];
    nSurf = subArray[1].size[2]*subArray[1].size[3];
    nInd = 1:nSurf;

    # applying the local solves (for loop)
    v1 =  [applyBlockOperator(subArray[ii],u0[ii],u1[ii],  uN[ii],  uNp[ii]) for ii = 1:nLayer ];

    Lu = zeros(Complex{Float64},2*(nLayer-1)*nSurf);

    jj = 1;

    Lu[nInd + (2*jj-2)*nSurf]  = vec(v1[jj][3]) - uN[jj];
    Lu[nInd + (2*jj-1)*nSurf]  = vec(v1[jj][4])  ;

    for ii=2:nLayer-1
        Lu[nInd + (2*ii-2)*nSurf] = vec(v1[ii][3]) - uN[ii];
        Lu[nInd + (2*ii-1)*nSurf] = vec(v1[ii][4]) ;
    end
    return Lu
end

# code version with a preallocated tuple
function devectorizeBdyData(subArray, uGamma)
    # function to tranform the vectorial uGamma in to a set of 4 arrays for an easier
    # evaluation of the integral operators
    v0  = [];
    v1  = [];
    vN  = [];
    vNp = [];
    # obtaining the number of layers
    nLayer = size(subArray)[1];
    nSurf = subArray[1].size[2]*subArray[1].size[3];
    nInd = 1:nSurf;

    for ii = 1:nLayer
        if ii == 1
            # extract the good traces and set to zero the other
            push!(v0,  0*uGamma[nInd]);
            push!(v1,  0*uGamma[nInd]);
            push!(vN,  uGamma[nInd]);
            push!(vNp, uGamma[nInd+nSurf]);
        elseif ii == (nLayer)
            #extract the good traces and put the rest to zero
            push!(v0,  uGamma[nInd+(2*ii-4)*nSurf]);
            push!(v1,  uGamma[nInd+(2*ii-3)*nSurf]);
            push!(vN,  0*uGamma[nInd+(2*ii-4)*nSurf]);
            push!(vNp, 0*uGamma[nInd+(2*ii-3)*nSurf]);
        else
            # fill the rest of the arrays with the data
            push!(v0,  uGamma[nInd+(2*ii-4)*nSurf]);
            push!(v1,  uGamma[nInd+(2*ii-3)*nSurf]);
            push!(vN,  uGamma[nInd+(2*ii-2)*nSurf]);
            push!(vNp, uGamma[nInd+(2*ii-1)*nSurf]);
        end
    end
    return (v0,v1,vN,vNp )

end

 #to be DONE, in order to have everything nicely encapsulated
function sourcePartition(f, nx::Int,ny::Int,nz::Int, npml::Int,nzi, nLayer )
    # partitioning the source % TODO make it a function
    ff = [];
    nzd = nzi+2*npml;
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

    return ff
end

function concatenate(uSolArray, nx::Int, ny::Int, nz::Int, npml::Int, nzi, nLayer)

    uSol = zeros(Complex{Float64},nx,ny,nz);
    uSol[:,:,1:(nzi+npml)] = reshape(uSolArray[1],nx,ny,nzi+2*npml)[:,:,1:(nzi+npml)];

    for ii = 1:nLayer-2
        uSol[:,:,npml+(1:nzi)+ii*nzi] =  reshape(uSolArray[ii+1],nx,ny,nzi+2*npml)[:,:,npml+(1:nzi)];
    end

    ii = nLayer-1;
    uSol[:,:,npml+(1:(nzi+npml))+ii*nzi] =  reshape(uSolArray[ii+1],nx,ny,nzi+2*npml)[:,:,npml+(1:(nzi+npml))];

    return uSol
end

function generatePermutationMatrix(nx,ny,nLayer )
    nInd = 1:nx*ny;
    nSurf = nx*ny;
    E = speye(4*(nLayer-1));
    p_aux   = kron(linspace(1, 2*(nLayer-1)-1, nLayer-1 ).', [1 1]) + kron(ones(1,nLayer-1), [0 2*(nLayer-1) ]);
    p_aux_2 = kron(linspace(2, 2*(nLayer-1) , nLayer-1 ).', [1 1]) + kron(ones(1,nLayer-1), [2*(nLayer-1) 0]);
    p = E[vec(hcat(round(Int64,p_aux), round(Int64,p_aux_2))),: ];
    P = kron(p, speye(nSurf));
    return P;
end

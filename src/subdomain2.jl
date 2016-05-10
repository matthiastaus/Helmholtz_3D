# Subdomain type
# the main idea behind is to abstract all the information needed to perform the
# method of polarized traces.

include("HelmholtzMatrix.jl");
include("model.jl")

type Subdomain
    model
    ind_0::Array{Int64,1}
    ind_1::Array{Int64,1}
    ind_n::Array{Int64,1}
    ind_np::Array{Int64,1}
    indVolInt::Array{Int64,1}
    indVol::Array{Int64,1}
    indVolIntLocal::Array{Int64,1}

    function Subdomain(model::Model, dummy)
        # m = matrix(nx,ny,nz)
        (ind_0, ind_1, ind_n , ind_np) = extractBoundaryIndices(model);
        indVolInt = extractVolIntIndices(model);
        indVol    = extractVolIndices(model);
        indVolIntLocal = extractVolIntLocalIndices(model);
        new(model, ind_0[:], ind_1[:], ind_n[:], ind_np[:], indVolInt[:], indVol[:], indVolIntLocal[:]) # don't know if it's the best answer
    end
end

function solve(subdomain::Subdomain, f::Array{Complex128,1})
    # u = solve(subdomain::Subdomain, f)
    # function that solves the system Hu=f in the subdomain
    u = solve(subdomain.model, f[:]);
end

function solve(subdomain::Subdomain, f::Array{Complex128,2})
    # u = solve(subdomain::Subdomain, f)
    # function that solves the system Hu=f in the subdomain
    u = solve(subdomain.model, f);
end

function extractBoundaryData(subdomain::Subdomain, u)
    # Function to extract the boundary data from a solution u
    # input   subdomain: subdomain associated to the solution
    #         u        : solution
    # output  (u0, u1, uN, uNp) : tuple of the solution at different depth
    # check size
    if (size(u[:])[1] == subdomain.model.size[1])
        u0  = u[subdomain.ind_0];
        u1  = u[subdomain.ind_1];
        uN  = u[subdomain.ind_n];
        uNp = u[subdomain.ind_np];
        return (u0,u1,uN,uNp)
    else
        print("Dimension mismatch \n");
        return 0
    end
end


function applyBlockOperator(subdomain::Subdomain,v0::Array{Complex128,1},v1::Array{Complex128,1},
                            vN::Array{Complex128,1},vNp::Array{Complex128,1})
    # function to apply the local matricial operator to the interface data
    # and we sample it at the interface
    # allocating the source
    f = zeros(Complex{Float64},subdomain.model.size[1])[:];
    # filling the source with the correct single and double layer potentials
    # TO DO: put the correct operators in here (we need to multiply the traces)
    # by the correct weights
    if length(subdomain.ind_1)>0
        # println("not the top layer")
        # println(size(v0))
        f[subdomain.ind_1 ] = -subdomain.model.H[subdomain.ind_1,subdomain.ind_0]*v0;
        f[subdomain.ind_0 ] =  subdomain.model.H[subdomain.ind_0,subdomain.ind_1]*v1;
    end
    if length(subdomain.ind_n)>0
        # println("not the bottom layer")
        # println(size(vN))
        f[subdomain.ind_np] =  subdomain.model.H[subdomain.ind_np,subdomain.ind_n]*vN;
        f[subdomain.ind_n ] = -subdomain.model.H[subdomain.ind_n,subdomain.ind_np]*vNp;
    end

    u = solve(subdomain, f[:]);

    u0  = u[subdomain.ind_0 ];
    u1  = u[subdomain.ind_1 ];
    uN  = u[subdomain.ind_n ];
    uNp = u[subdomain.ind_np];
    return (u0,u1,uN,uNp)

end

# version of the block operator to act for multiple right-hand side
function applyBlockOperator(subdomain::Subdomain, v0::Array{Complex128,2},v1::Array{Complex128,2},
                            vN::Array{Complex128,2},vNp::Array{Complex128,2})
    # function to apply the local matricial operator to the interface data
    # and we sample it at the interface
    nrhs =  size(v0)[2]
    # allocating the source
    f = zeros(Complex{Float64},subdomain.model.size[1],nrhs);
    # filling the source with the correct single and double layer potentials
    # TO DO: put the correct operators in here (we need to multiply the traces)
    # by the correct weights
    for ii = 1:nrhs
        if length(subdomain.ind_1)>0
            f[subdomain.ind_1,ii ] = -subdomain.model.H[subdomain.ind_1,subdomain.ind_0]*v0[:,ii];
            f[subdomain.ind_0,ii ] =  subdomain.model.H[subdomain.ind_0,subdomain.ind_1]*v1[:,ii];
        end
        if length(subdomain.ind_n)>0
            f[subdomain.ind_np,ii] =  subdomain.model.H[subdomain.ind_np,subdomain.ind_n]*vN[:,ii];
            f[subdomain.ind_n ,ii] = -subdomain.model.H[subdomain.ind_n,subdomain.ind_np]*vNp[:,ii];
        end
    end
    u = solve(subdomain, f);

    U   = reshape(u,subdomain.model.size[1], nrhs)
    # extracting the traces at the boundaries
    u0  = U[subdomain.ind_0 ,:];
    u1  = U[subdomain.ind_1 ,:];
    uN  = U[subdomain.ind_n ,:];
    uNp = U[subdomain.ind_np,:];

    return (u0,u1,uN,uNp)

end



function solveLocal(subDomains, source::Array{Complex128,1})
    # function that solves all the local problems given a globally
    # defined source

    println("Solving everything locally")
    # partition the sources
    sourceArray = sourcePartition(subDomains, source)

    # number of subdomains
    nSubs = length(subDomains);

    # Solve all the local problems
    uLocalArray = [solve(subDomains[ii],sourceArray[ii] )  for  ii = 1:nSubs];

    return uLocalArray
end

function sourcePartition(subDomains, source::Array{Complex128,1})

    println("partitioning the source")
    # partitioning the source % TODO make it a function
    nSubs = length(subDomains);

    n = length(subDomains[1].model.x)*length(subDomains[1].model.y)
    # building the local rhs
    rhsLocal = [ zeros(Complex128,subDomains[ii].model.size[1]) for ii = 1:nSubs ]

    # copying the wave-fields
    for ii = 1:nSubs
        rhsLocal[ii][subDomains[ii].indVolIntLocal] = source[subDomains[ii].indVolInt]
    end
    return rhsLocal
end


function extractFullBoundaryData(subDomains, uLocalArray)
    # Function to extract the boundary data from an array of local solutions
    # input   SubArray: subdomain associated to the solution
    #         u        : solution
    # output  (u0, u1, uN, uNp) : tuple of the solution at different depth
    # check size

    # partitioning the source % TODO make it a function
    nSubs = length(subDomains);

    n = subDomains[1].model.size[2]*subDomains[1].model.size[3];

    u_0  = zeros(Complex128,n*nSubs)
    u_1  = zeros(Complex128,n*nSubs)
    u_n  = zeros(Complex128,n*nSubs)
    u_np = zeros(Complex128,n*nSubs)

    index = 1:n

    # populate the traces

    for ii = 1:nSubs
        # this will be slow most likely but it is readable
        # we will need to modify this part
        ind_0  = subDomains[ii].ind_0
        ind_1  = subDomains[ii].ind_1
        ind_n  = subDomains[ii].ind_n
        ind_np = subDomains[ii].ind_np

        if ii != 1
            u_0[(ii-1)*n + index] = uLocalArray[ii][ind_0]
            u_1[(ii-1)*n + index] = uLocalArray[ii][ind_1]
        end
        if ii !=nSubs
            u_n[(ii-1)*n  + index] = uLocalArray[ii][ind_n]
            u_np[(ii-1)*n + index] = uLocalArray[ii][ind_np]
        end

    end

    return (u_0, u_1, u_n, u_np)

end

function extractRHS(subDomains,source::Array{Complex128,1})
    #function to produce and extra the rhs

    uLocalArray = solveLocal(subDomains, source)

    return extractFullBoundaryData(subDomains, uLocalArray)
end


function devectorizeBdyData(subArray, uGamma)
    # function to tranform the vectorial uGamma in to a set of 4 arrays for an easier
    # evaluation of the integral operators
    v0  = [];
    v1  = [];
    vN  = [];
    vNp = [];
    # obtaining the number of layers
    nLayer = size(subArray)[1];
    nSurf = length(subArray[1].model.x)*length(subArray[1].model.y);
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
    return (v0,v1,vN,vNp)
end

function devectorizeBdyDataContiguous(subArray, uGamma)
    # function to tranform the vectorial uGamma in to a set of 4 arrays for an easier
    # evaluation of the integral operators

    # obtaining the number of layers
    nLayer = size(subArray)[1];
    nSurf = subArray[1].model.size[2]*subArray[1].model.size[3];
    nInd = 1:nSurf;

    v0  = zeros(Complex128, nSurf*nLayer);
    v1  = zeros(Complex128, nSurf*nLayer);
    vN  = zeros(Complex128, nSurf*nLayer);
    vNp = zeros(Complex128, nSurf*nLayer);

    for ii = 1:nLayer
        if ii == 1
            # extract the good traces and set to zero the other
            v0[nInd]  = 0*uGamma[nInd];
            v1[nInd]  = 0*uGamma[nInd];
            vN[nInd]  = uGamma[nInd];
            vNp[nInd] = uGamma[nInd+nSurf];
        elseif ii == (nLayer)
            #extract the good traces and put the rest to zero
            v0[nInd+ (ii-1)*nSurf] =   uGamma[nInd+(2*ii-4)*nSurf];
            v1[nInd+ (ii-1)*nSurf] =   uGamma[nInd+(2*ii-3)*nSurf];
            vN[nInd+ (ii-1)*nSurf] =  0*uGamma[nInd+(2*ii-4)*nSurf];
            vNp[nInd+ (ii-1)*nSurf] = 0*uGamma[nInd+(2*ii-3)*nSurf];
        else
            # fill the rest of the arrays with the data
            v0[nInd+ (ii-1)*nSurf]  =   uGamma[nInd+(2*ii-4)*nSurf];
            v1[nInd+ (ii-1)*nSurf]  =   uGamma[nInd+(2*ii-3)*nSurf];
            vN[nInd+ (ii-1)*nSurf]  =   uGamma[nInd+(2*ii-2)*nSurf];
            vNp[nInd+ (ii-1)*nSurf] =   uGamma[nInd+(2*ii-1)*nSurf];
        end
    end
    return (v0,v1,vN,vNp)
end


function vectorizeBdyData(subDomains, uBdyData)
    # function to take the output of extract Boundary data and put it in vectorized form
    nSubs = length(subDomains);
    n = subDomains[1].model.size[2]*subDomains[1].model.size[3]

    uBdy = zeros(Complex{Float64},2*(nSubs-1)*n);
    nInd = 1:n;

    uBdy[nInd] = uBdyData[3][nInd];
    for ii = 1:nSubs-2
        uBdy[nInd+(2*ii-1)*n] = uBdyData[2][(ii*n +nInd)];
        uBdy[nInd+2*ii*n]     = uBdyData[3][(ii*n +nInd)];
    end
    uBdy[nInd+(2*nSubs-3)*n] = uBdyData[2][(nSubs-1)*n+nInd];

    return uBdy
end


function vectorizePolarizedBdyDataRHS(subDomains,uBdyData)
    # function to take the output of extract Boundary data and put it in vectorized form

    nSubs = length(subDomains);
    n = length(subDomains[1].model.x)*length(subDomains[1].model.y)

    f1 = zeros(Complex{Float64},2*(nSubs-1)*n);
    nInd = 1:n;

    f1[nInd] = uBdyData[3][nInd];
    for ii = 1:nSubs-2
        f1[nInd+(2*ii-1)*n] = uBdyData[2][(ii*n +nInd)];
        f1[nInd+2*ii*n]     = uBdyData[3][(ii*n +nInd)];
    end
    f1[nInd+(2*nSubs-3)*n] = uBdyData[2][(nSubs-1)*n+nInd];

    f0 = zeros(Complex{Float64},2*(nSubs-1)*n);
    nInd = 1:n;

    f0[nInd] =  uBdyData[4][nInd];
    for ii = 1:nSubs-2
        f0[nInd+(2*ii-1)*n] = uBdyData[1][ii*n+ nInd];
        f0[nInd+2*ii*n]     = uBdyData[4][ii*n+ nInd];
    end
    f0[nInd+(2*nSubs-3)*n] = uBdyData[1][(nSubs-1)*n+nInd];

    return vcat(f1,f0)
end

function vectorizePolarizedBdyData(subDomains,uBdyData)
    # function to take the output of extract Boundary data and put it in vectorized form
    nSubs = length(subDomains);
    n = length(subDomains[1].model.x)*length(subDomains[1].model.y)

    uBdy = zeros(Complex{Float64},4*(nSubs-1)*n);

    nInd = 1:n;

    uBdy[nInd]   = uBdyData[3][nInd];
    uBdy[nInd+n] = uBdyData[4][nInd];

    for ii = 1:nSubs-2
        uBdy[nInd+(4*ii-2)*n] = uBdyData[1][ii*n+ nInd];
        uBdy[nInd+(4*ii-1)*n] = uBdyData[2][ii*n+ nInd];
        uBdy[nInd+(4*ii  )*n] = uBdyData[3][ii*n+ nInd];
        uBdy[nInd+(4*ii+1)*n] = uBdyData[4][ii*n+ nInd];
    end
    ii = nSubs-1
    uBdy[nInd+(4*ii-2)*n] = uBdyData[1][(ii*n+nInd)];
    uBdy[nInd+(4*ii-1)*n] = uBdyData[2][(ii*n+nInd)];

    return uBdy
end


function applyM(SubArray, uGamma)
    # function to apply M to uGamma

    # decomposing the uGamma to be suitable for a condensed application
    (v0,v1,vN,vNp ) = devectorizeBdyData(SubArray, uGamma);

    # applying
    nLayer = length(SubArray)
    # TODO this has to be done in parallel using a list of RemoteRefs
    Au = [ applyBlockOperator(SubArray[ii],v0[ii],v1[ii],vN[ii],vNp[ii]) for ii = 1:nLayer ];

    # it need to be a vector
    u = vectorizeBdyDataV2(Au);

    Mu = u - uGamma;

    return Mu[:]

end


function vectorizeBdyDataV2(uBdyData)
    # function to take the output of extract Boundary data and put it in vectorized form
    nLayer = length(uBdyData)
    nn     = length(uBdyData[1][3])

    uBdy = zeros(Complex{Float64},2*(nLayer-1)*nn);

    nInd = 1:nn;

    uBdy[nInd] = uBdyData[1][3];
    for ii = 1:nLayer-2
        uBdy[nInd+(2*ii-1)*nn] = uBdyData[ii+1][2];
        uBdy[nInd+2*ii*nn] = uBdyData[ii+1][3];
    end
    uBdy[nInd+(2*nLayer-3)*nn] = uBdyData[nLayer][2];

    return uBdy[:]
end




# Function for the application of the different blocks of the integral system
function applyMup(subDomains, uGamma; verbose = false)
    # function to apply M to uGamma
    # input subDomains : an array of subdomains
    #       uGamma   : data on the boundaries in vector form
    verbose && println("applying M up")
    # convert uGamma in a suitable vector to be applied
    (u0,u1,uN,uNp) = devectorizeBdyData(subDomains, uGamma);

    # obtaining the number of layers
    nLayer = size(subDomains)[1];
    nSurf = subDomains[1].model.size[2]*subDomains[1].model.size[3]
    nInd = 1:nSurf;

    # applying this has to be done in parallel applying RemoteRefs
    v1 = [ applyBlockOperator(subDomains[ii],0*u0[ii],0*u1[ii],uN[ii],uNp[ii]) for ii = 1:nLayer ];
    vN = [ applyBlockOperator(subDomains[ii],  u0[ii],  u1[ii],uN[ii],uNp[ii]) for ii = 1:nLayer ];

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

function applyMdown(subDomains, uGamma; verbose = false)
    # function to apply M to uGamma
    # input subDomains : an array of subdomains
    #       uGamma   : data on the boundaries in vector form
    verbose && println("applying M down")
    # convert uGamma in a suitable vector to be applied
    (u0,u1,uN,uNp) = devectorizeBdyData(subDomains, uGamma);

    # obtaining the number of layers
    nLayer = size(subDomains)[1];
    nSurf = subDomains[1].model.size[2]*subDomains[1].model.size[3]
    nInd = 1:nSurf;

    verbose && println("applying local solves")
    # applying the local solves (for loop)
    v1 = [ applyBlockOperator(subDomains[ii],u0[ii],u1[ii],  uN[ii],  uNp[ii]) for ii = 1:nLayer ];
    vN = [ applyBlockOperator(subDomains[ii],u0[ii],u1[ii],0*uN[ii],0*uNp[ii]) for ii = 1:nLayer ];

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

function applyM0up(subDomains, uGamma; verbose = false)
    # function to apply M to uGamma
    # input subDomains : an array of subdomains
    #       uGamma   : data on the boundaries in vector form

    verbose && println("applying M0 up")
    # convert uGamma in a suitable vector to be applied
    (u0,u1,uN,uNp) = devectorizeBdyData(subDomains, uGamma);

    # obtaining the number of layers
    nLayer = size(subDomains)[1];
    nSurf = subDomains[1].model.size[2]*subDomains[1].model.size[3]
    nInd = 1:nSurf;

    # applying
    # remember  v[ii][1] = v0[ii], v[ii][2] = v1[ii],
    #           v[ii][3] = vN[ii], v[ii][4] = vNp[ii]
    v1 = [ applyBlockOperator(subDomains[ii],0*u0[ii],0*u1[ii],uN[ii],uNp[ii]) for ii = 1:nLayer ];
    vN = [ applyBlockOperator(subDomains[ii],  u0[ii],  u1[ii],uN[ii],uNp[ii]) for ii = 1:nLayer ];

    Mu = zeros(Complex{Float64},2*(nLayer-1)*nSurf);

    Mu[nInd] =  vec(vN[1][4]);

    for ii=1:nLayer-2
        Mu[nInd + (2*ii-1)*nSurf] = - u0[ii+1] + vec(v1[ii+1][1]);
        Mu[nInd + (2*ii  )*nSurf] =            + vec(vN[ii+1][4]);
    end
    ii = nLayer-1;
    Mu[nInd + (2*ii-1)*nSurf] = -  u0[ii+1] #+ vec(v1[ii+1][1]); # this last one has to be zero

    return Mu[:]

end

function applyM0down(subDomains, uGamma; verbose = false)
    # function to apply M to uGamma
    # input subDomains : an array of subdomains
    #       uGamma   : data on the boundaries in vector form

    verbose && println("applying M0 down")
    # convert uGamma in a suitable vector to be applied
    (u0,u1,uN,uNp) = devectorizeBdyData(subDomains, uGamma);

    # obtaining the number of layers
    nLayer = size(subDomains)[1];
    nSurf = subDomains[1].model.size[2]*subDomains[1].model.size[3]
    nInd = 1:nSurf;

    # applying the local solves (for loop)
    v1 = [ applyBlockOperator(subDomains[ii],u0[ii],u1[ii],  uN[ii],  uNp[ii]) for ii = 1:nLayer ];
    vN = [ applyBlockOperator(subDomains[ii],u0[ii],u1[ii],0*uN[ii],0*uNp[ii]) for ii = 1:nLayer ];

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

function applyMM(subDomains, uGammaPol; verbose = false)
    # function to apply the the polarized integral operator
    verbose && println("Applying the polarized matrix")
    uDown = uGammaPol[1:end/2];
    uUp = uGammaPol[(end/2+1):end];

    MMu = vcat(applyMdown( subDomains, uDown, verbose = verbose) + applyMup( subDomains, uUp, verbose = verbose),
               applyM0down(subDomains, uDown, verbose = verbose) + applyM0up(subDomains, uUp, verbose = verbose));
    return MMu;

end

## now the functions for the preconditioners

function applyDdown(subDomains, uGamma)
    # obtaining the number of layers
    nLayer = size(subDomains)[1];
    nSurf = subDomains[1].model.size[2]*subDomains[1].model.size[3]
    nInd    = 1:nSurf;

    dummyzero = zeros(Complex{Float64},nSurf);
    Du = zeros(Complex{Float64},2*(nLayer-1)*nSurf);

    Du[nInd]        = -uGamma[nInd];
    Du[nInd+nSurf]  = -uGamma[nInd+nSurf];

    for ii=1:nLayer-2
        uNm  = uGamma[nInd+ (2*ii-2)*nSurf];
        uNpm = uGamma[nInd+ (2*ii-1)*nSurf];
        uN   = uGamma[nInd+ (2*ii  )*nSurf];
        uNp  = uGamma[nInd+ (2*ii+1)*nSurf];

        (v0, v1, vN, vNp) = applyBlockOperator(subDomains[ii+1],uNm,uNpm, dummyzero,dummyzero);

        Du[nInd+ 2*ii*nSurf]      = vec(vN)  - vec(uN);
        Du[nInd+ (2*ii+1)*nSurf]  = vec(vNp) - vec(uNp);

    end

    return Du
end

function applyDinvDown(subDomains, uGamma)
    # obtaining the number of layers
    nLayer = size(subDomains)[1];
    nSurf = subDomains[1].model.size[2]*subDomains[1].model.size[3]
    nInd    = 1:nSurf;

    dummyzero = zeros(Complex{Float64},nSurf);
    Dinvu = zeros(Complex{Float64},2*(nLayer-1)*nSurf);

    Dinvu[nInd]        = -uGamma[nInd];
    Dinvu[nInd+nSurf]  = -uGamma[nInd+nSurf];
    vN   = Dinvu[nInd] ;
    vNp  = Dinvu[nInd+nSurf];

    for ii=1:nLayer-2
        uN = uGamma[nInd+ 2*ii*nSurf];
        uNp= uGamma[nInd+ (2*ii+1)*nSurf];

        (v0, v1, vN, vNp) = applyBlockOperator(subDomains[ii+1],vN,vNp, dummyzero,dummyzero);

        Dinvu[nInd+ 2*ii*nSurf]      = vec(vN)  - vec(uN);
        Dinvu[nInd+ (2*ii+1)*nSurf]  = vec(vNp) - vec(uNp);

        vN   = Dinvu[nInd+ 2*ii*nSurf] ;
        vNp  = Dinvu[nInd+ (2*ii+1)*nSurf];
    end

    return Dinvu
end


function applyDup(subDomains, uGamma)
    # obtaining the number of layers
    nLayer = size(subDomains)[1];
    nSurf = subDomains[1].model.size[2]*subDomains[1].model.size[3]
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

        (v0, v1, vN, vNp) = applyBlockOperator(subDomains[ii+1],dummyzero,dummyzero,u0m,u1m);

        Dup[nInd+ (2*ii-2)*nSurf]  = vec(v0) - vec(u0);
        Dup[nInd+ (2*ii-1)*nSurf]  = vec(v1) - vec(u1);

    end

    return Dup
end

function applyDinvUp(subDomains, uGamma)
    # obtaining the number of layers
    nLayer = size(subDomains)[1];
    nSurf = subDomains[1].model.size[2]*subDomains[1].model.size[3]
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

        (v0, v1, vN, vNp) = applyBlockOperator(subDomains[ii+1],dummyzero,dummyzero,v0,v1);

        Dinvu[nInd+ (2*ii-2)*nSurf]  = vec(v0) - vec(u0);
        Dinvu[nInd+ (2*ii-1)*nSurf]  = vec(v1) - vec(u1);

        v0  = Dinvu[nInd+ (2*ii-2)*nSurf] ;
        v1  = Dinvu[nInd+ (2*ii-1)*nSurf];
    end

    return Dinvu
end


function applyU(subDomains, uGamma)
    # function to apply M to uGamma
    # input subDomains : an array of subdomains
    #       uGamma   : data on the boundaries in vector form

    # convert uGamma in a suitable vector to be applied
    (u0,u1,uN,uNp) = devectorizeBdyData(subDomains, uGamma);

    # obtaining the number of layers
    nLayer = size(subDomains)[1];
    nSurf = subDomains[1].model.size[2]*subDomains[1].model.size[3];
    nInd = 1:nSurf;

    # applying the local solves (for loop)
    v1 = [ applyBlockOperator(subDomains[ii],u0[ii],u1[ii],  uN[ii],  uNp[ii]) for ii = 1:nLayer ];

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


function applyL(subDomains, uGamma)
    # function to apply M to uGamma
    # input subDomains : an array of subdomains
    #       uGamma   : data on the boundaries in vector form

    # convert uGamma in a suitable vector to be applied
    (u0,u1,uN,uNp) = devectorizeBdyData(subDomains, uGamma);

    # obtaining the number of layers
    nLayer = size(subDomains)[1];
    nSurf = subDomains[1].model.size[2]*subDomains[1].model.size[3] ;
    nInd = 1:nSurf;

    # applying the local solves (for loop)
    v1 = [ applyBlockOperator(subDomains[ii],u0[ii],u1[ii],  uN[ii],  uNp[ii]) for ii = 1:nLayer ];

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

##### miscelaneous functions


function generatePermutationMatrix(n::Int64,nSubs::Int64 )
    nInd = 1:n;
    nSurf = n;
    E = speye(4*(nSubs-1));
    p_aux   = kron(linspace(1, 2*(nSubs-1)-1, nSubs-1 ).', [1 1]) + kron(ones(1,nSubs-1), [0 2*(nSubs-1) ]);
    p_aux_2 = kron(linspace(2, 2*(nSubs-1) , nSubs-1 ).', [1 1]) + kron(ones(1,nSubs-1), [2*(nSubs-1) 0]);
    p = E[vec(hcat(round(Int64,p_aux), round(Int64,p_aux_2))),: ];
    P = kron(p, speye(nSurf));
    return P;
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



#####################################################################
#                                                                   #
#          Optimized Functions                                      #
#                                                                   #
#####################################################################


# Function for the application of the different blocks of the integral system
# This is an optimized version that uses only 4 solves per layer instead of 8
# moreover, it allocates less memory
function applyMMOpt(subDomains, uGamma)
    # function to apply M to uGamma
    # input subDomains : an array of subdomains
    #       uGamma   : data on the boundaries in vector form

    uDown = uGamma[1:round(Integer, end/2)]
    uUp   = uGamma[1+round(Integer, end/2):end]
    # convert uGamma in a suitable vector to be applied
    (u0Down,u1Down,uNDown,uNpDown) = devectorizeBdyData(subDomains, uDown);
    (u0Up  ,u1Up  ,uNUp  ,uNpUp)   = devectorizeBdyData(subDomains, uUp);

    # obtaining the number of layers
    nLayer = size(subDomains)[1];
    nSurf = subDomains[1].model.size[2]*subDomains[1].model.size[3]
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



# Function for the application of the different blocks of the integral system
# This is an optimized version that uses only 1 solve per layer instead of 8
# It feeds multiple right hand sides for each solve. The main idea is to take
# advantage to BLAS 3 operations. Unfortunately, UMFPACK does not take advantage
# of the last. However, Pardiso MKL does, in which case we can observe a good
# speed up.
function applyMMOptUmf(subDomains, uGamma)
    # function to apply M to uGamma using an application via multiple right-hand sides
    # input subDomains : an array of subdomains
    #       uGamma   : data on the boundaries in vector form

    uDown = uGamma[1:round(Integer, end/2)]
    uUp   = uGamma[1+round(Integer, end/2):end]
    # convert uGamma in a suitable vector to be applied
    (u0Down,u1Down,uNDown,uNpDown) = devectorizeBdyData(subDomains, uDown);
    (u0Up  ,u1Up  ,uNUp  ,uNpUp)   = devectorizeBdyData(subDomains, uUp);

    # obtaining the number of layers
    nLayer = size(subDomains)[1];
    nSurf = subDomains[1].model.size[2]*subDomains[1].model.size[3]
    nInd = 1:nSurf;

    # In this case we use only one solve
    # but using multiple right-hand sides in order to fully take advantage of the BLAS3
    # routines in MUMPS and MKLPardiso
    V = [ applyBlockOperator(subDomains[ii],
            hcat(u0Down[ii], u0Down[ii]+u0Up[ii], u0Down[ii], u0Up[ii]+u0Down[ii]),
            hcat(u1Down[ii], u1Down[ii]+u1Up[ii], u1Down[ii], u1Down[ii]+u1Up[ii]),
            hcat(uNUp[ii]+uNDown[ii], uNUp[ii], uNUp[ii]+uNDown[ii], uNUp[ii]),
            hcat(uNpUp[ii]+uNpDown[ii], uNpUp[ii], uNpUp[ii]+uNpDown[ii], uNpUp[ii] )) for ii = 1:nLayer ];


    Mu1 = zeros(Complex{Float64},2*(nLayer-1)*nSurf);

    Mu1[nInd] = - uNUp[1] - uNDown[1] + vec(V[1][3][:,2]);

    for ii=1:nLayer-2
        Mu1[nInd + (2*ii-1)*nSurf] = - u1Up[ii+1] - u1Down[ii+1] + vec(V[ii+1][2][:,1]);
        Mu1[nInd + (2*ii  )*nSurf] = - uNUp[ii+1] - uNDown[ii+1] + vec(V[ii+1][3][:,2]);
    end
    ii = nLayer-1;
    Mu1[nInd + (2*ii-1)*nSurf] = - u1Up[ii+1] - u1Down[ii+1] + vec(V[ii+1][2][:,1]) ;


    Mu = zeros(Complex{Float64},2*(nLayer-1)*nSurf);


    Mu[nInd] =  vec(V[1][4][:,4])- uNpDown[1] ;

    for ii=1:nLayer-2
        Mu[nInd + (2*ii-1)*nSurf] = - u0Up[ii+1] + vec(V[ii+1][1][:,3]);
        Mu[nInd + (2*ii  )*nSurf] = - uNpDown[ii+1] + vec(V[ii+1][4][:,4]);
    end
    ii = nLayer-1;
    Mu[nInd + (2*ii-1)*nSurf] = -  u0Up[ii+1] + vec(V[ii+1][1][:,3])

    return vcat(Mu1,Mu)

end

# Function for the application of the different blocks of the integral system
# This is an optimized version that uses only 1 solve per layer instead of 8
# it uses a contiguous organization of the data, in order to minimize cache misses
function applyMMOpt2(subDomains, uGamma)
    # function to apply M to uGamma using an application via multiple right-hand sides
    # input subDomains : an array of subdomains
    #       uGamma   : data on the boundaries in vector form

    uDown = uGamma[1:round(Integer, end/2)]
    uUp   = uGamma[1+round(Integer, end/2):end]
    # convert uGamma in a suitable vector to be applied
    (u0Down,u1Down,uNDown,uNpDown) = devectorizeBdyDataContiguous(subDomains, uDown);
    (u0Up  ,u1Up  ,uNUp  ,uNpUp)   = devectorizeBdyDataContiguous(subDomains, uUp);

    # obtaining the number of layers
    nLayer = size(subDomains)[1];
    nSurf = subDomains[1].model.size[2]*subDomains[1].model.size[3]
    nInd = 1:nSurf;

    # In this case we use only one solve
    # but using multiple right-hand sides in order to fully take advantage of the BLAS3
    # routines in MUMPS and MKLPardiso
    V = [ applyBlockOperator(subDomains[ii],
            hcat(u0Down[(ii-1)*nSurf + nInd], u0Down[(ii-1)*nSurf + nInd]+u0Up[(ii-1)*nSurf + nInd], u0Down[(ii-1)*nSurf + nInd], u0Up[(ii-1)*nSurf + nInd]+u0Down[(ii-1)*nSurf + nInd]),
            hcat(u1Down[(ii-1)*nSurf + nInd], u1Down[(ii-1)*nSurf + nInd]+u1Up[(ii-1)*nSurf + nInd], u1Down[(ii-1)*nSurf + nInd], u1Down[(ii-1)*nSurf + nInd]+u1Up[(ii-1)*nSurf + nInd]),
            hcat(uNUp[(ii-1)*nSurf + nInd]+uNDown[(ii-1)*nSurf + nInd], uNUp[(ii-1)*nSurf + nInd], uNUp[(ii-1)*nSurf + nInd]+uNDown[(ii-1)*nSurf + nInd], uNUp[(ii-1)*nSurf + nInd]),
            hcat(uNpUp[(ii-1)*nSurf + nInd]+uNpDown[(ii-1)*nSurf + nInd], uNpUp[(ii-1)*nSurf + nInd], uNpUp[(ii-1)*nSurf + nInd]+uNpDown[(ii-1)*nSurf + nInd], uNpUp[(ii-1)*nSurf + nInd] )) for ii = 1:nLayer ];

    # We extract the boundary data from V using only one for loop
    # to minimize cache misses 
    Mu1 = zeros(Complex{Float64},2*(nLayer-1)*nSurf);
    Mu = zeros(Complex{Float64},2*(nLayer-1)*nSurf);

    ii = 1;
    Mu1[nInd] = - uNUp[(ii-1)*nSurf + nInd] - uNDown[(ii-1)*nSurf + nInd] + vec(V[1][3][:,2]);
    Mu[nInd]  =  vec(V[1][4][:,4])- uNpDown[(ii-1)*nSurf + nInd] ;

    for ii=1:nLayer-2
        Mu1[nInd + (2*ii-1)*nSurf] = - u1Up[(ii)*nSurf + nInd] - u1Down[(ii)*nSurf + nInd] + vec(V[ii+1][2][:,1]);
        Mu1[nInd + (2*ii  )*nSurf] = - uNUp[(ii)*nSurf + nInd] - uNDown[(ii)*nSurf + nInd] + vec(V[ii+1][3][:,2]);

        Mu[nInd + (2*ii-1)*nSurf] = - u0Up[(ii)*nSurf + nInd] + vec(V[ii+1][1][:,3]);
        Mu[nInd + (2*ii  )*nSurf] = - uNpDown[(ii)*nSurf + nInd] + vec(V[ii+1][4][:,4]);
    end
    ii = nLayer-1;
    Mu1[nInd + (2*ii-1)*nSurf] = - u1Up[(ii)*nSurf + nInd] - u1Down[(ii)*nSurf + nInd] + vec(V[ii+1][2][:,1]) ;
    Mu[nInd + (2*ii-1)*nSurf]  = -  u0Up[(ii)*nSurf + nInd] + vec(V[ii+1][1][:,3])

    return vcat(Mu1,Mu)

end





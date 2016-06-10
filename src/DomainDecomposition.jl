# class to encapsulate the Domain decomposition
#

include("subdomain2.jl");

type DomainDecomposition
    # array of pointer to subdomains
    subDomains::Array{Subdomain,1}
    LimsN::Array{Int64,2}
    Lims1::Array{Int64,2}


    function DomainDecomposition(subDomains::Array{Subdomain,1}, dummy::Int64 )
        nLayers = length(subDomains)
        LimsN = zeros(Int64, nLayer-1,2)
        Lims1 = zeros(Int64, nLayer-1,2)
        LimsN[1,:] = [1,length(subDomains[1].ind_n)];
        Lims1[1,:] = LimsN[1,:] +  length(subDomains[2].ind_1);

        for ii = 2:nLayer-1
            LimsN[ii,:] = Lims1[ii-1,:] + length(subDomains[ii].ind_n);
            Lims1[ii,:] = LimsN[ii,:]   + length(subDomains[ii+1].ind_1);
        end

        new(subDomains,LimsN, Lims1 )
    end
end

function applyMMOptDDM(DDM::DomainDecomposition, uGamma)
    # function to apply M to uGamma
    # using the DDM class
    #  we use a Limits vector in order to handle subdomains
    # with a different number of boundary points
    # input subDomains : an array of subdomains
    #       uGamma   : data on the boundaries in vector form

    uDown = uGamma[1:round(Integer, end/2)]
    uUp   = uGamma[1+round(Integer, end/2):end]
    # convert uGamma in a suitable vector to be applied
    (u0Down,u1Down,uNDown,uNpDown) = devectorizeBdyData(DDM.subDomains, uDown);
    (u0Up  ,u1Up  ,uNUp  ,uNpUp)   = devectorizeBdyData(DDM.subDomains, uUp);

    # obtaining the number of layers
    nLayer = size(DDM.subDomains)[1];
    nSurf = DDM.subDomains[1].model.size[2]*DDM.subDomains[1].model.size[3]
    nInd = 1:nSurf;

    # building the Limits vectors in order to delimite the entries in the
    # vector containing the application of the integral operator
    Lims1 = DDM.LimsN;
    Lims2 = DDM.Lims1;

    #TODO modify this functio in order to solve the three rhs in one shot
    # applying this has to be done in parallel applying RemoteRefs
    v1 = [ applyBlockOperator(DDM.subDomains[ii],u0Down[ii]         ,u1Down[ii]         ,
                              uNUp[ii]+uNDown[ii],uNpUp[ii]+uNpDown[ii]) for ii = 1:nLayer ];
    vN = [ applyBlockOperator(DDM.subDomains[ii],u0Down[ii]+u0Up[ii],u1Down[ii]+u1Up[ii],
                              uNUp[ii]           ,uNpUp[ii]            ) for ii = 1:nLayer ];

    Mu1 = zeros(Complex{Float64},round(Integer,maximum(Lims2)) );

    Mu1[Lims1[1,1]:Lims1[1,2]] = - uNUp[1] - uNDown[1] + vec(vN[1][3]);

    for ii=1:nLayer-2
        Mu1[Lims2[ii,1]:Lims2[ii,2]]     =  - u1Up[ii+1] - u1Down[ii+1] + vec(v1[ii+1][2]);
        Mu1[Lims1[ii+1,1]:Lims1[ii+1,2]] = - uNUp[ii+1] - uNDown[ii+1] + vec(vN[ii+1][3]);
    end
    ii = nLayer-1;
    Mu1[Lims2[ii,1]:Lims2[ii,2]] = - u1Up[ii+1] - u1Down[ii+1] + vec(v1[ii+1][2]) ;


    v1 = [ applyBlockOperator(DDM.subDomains[ii],u0Down[ii],u1Down[ii],uNUp[ii]+uNDown[ii],uNpUp[ii]+uNpDown[ii]) for ii = 1:nLayer ];
    vN = [ applyBlockOperator(DDM.subDomains[ii],u0Up[ii]+u0Down[ii],u1Down[ii]+u1Up[ii],uNUp[ii],uNpUp[ii]) for ii = 1:nLayer ];

    Mu = zeros(Complex{Float64},round(Integer,maximum(Lims2)) );


    Mu[Lims1[1,1]:Lims1[1,2]] =  vec(vN[1][4])- uNpDown[1] ;

    for ii=1:nLayer-2
        Mu[Lims2[ii,1]:Lims2[ii,2]] = - u0Up[ii+1] + vec(v1[ii+1][1]);
        Mu[Lims1[ii+1,1]:Lims1[ii+1,2]] = - uNpDown[ii+1] + vec(vN[ii+1][4]);
    end
    ii = nLayer-1;
    Mu[Lims2[ii,1]:Lims2[ii,2]] = -  u0Up[ii+1] + vec(v1[ii+1][1])

    return vcat(Mu1,Mu)

end

function applyMMOptMultDMM(DDM::DomainDecomposition, uGamma::Array{Complex128,1})
    # function to apply M to uGamma using an application via multiple right-hand sides
    # input subDomains : an array of subdomains
    #       uGamma   : data on the boundaries in vector form

    uDown = uGamma[1:round(Integer, end/2)]
    uUp   = uGamma[1+round(Integer, end/2):end]
    # convert uGamma in a suitable vector to be applied
    (u0Down,u1Down,uNDown,uNpDown) = devectorizeBdyData(DDM.subDomains, uDown);
    (u0Up  ,u1Up  ,uNUp  ,uNpUp)   = devectorizeBdyData(DDM.subDomains, uUp);

    # obtaining the number of layers
    nLayer = size(DDM.subDomains)[1];
    nSurf = DDM.subDomains[1].model.size[2]*DDM.subDomains[1].model.size[3]
    nInd = 1:nSurf;

    # In this case we use only one solve
    # but using multiple right-hand sides in order to fully take advantage of the BLAS3
    # routines in MUMPS and MKLPardiso
    V = [ applyBlockOperator(DDM.subDomains[ii],
            hcat(u0Down[ii]           , u0Down[ii]+u0Up[ii], u0Down[ii]           , u0Up[ii]+u0Down[ii]),
            hcat(u1Down[ii]           , u1Down[ii]+u1Up[ii], u1Down[ii]           , u1Down[ii]+u1Up[ii]),
            hcat(uNUp[ii]+uNDown[ii]  , uNUp[ii]           , uNUp[ii]+uNDown[ii]  , uNUp[ii]),
            hcat(uNpUp[ii]+uNpDown[ii], uNpUp[ii]          , uNpUp[ii]+uNpDown[ii], uNpUp[ii] )) for ii = 1:nLayer ];


    Lims1 = DDM.LimsN;
    Lims2 = DDM.Lims1;

    Mu1 = zeros(Complex{Float64},round(Integer,maximum(Lims2)) );

    Mu1[Lims1[1,1]:Lims1[1,2]] = - uNUp[1] - uNDown[1] + vec(V[1][3][:,2]);

    for ii=1:nLayer-2
        Mu1[Lims2[ii,1]:Lims2[ii,2]] = - u1Up[ii+1] - u1Down[ii+1] + vec(V[ii+1][2][:,1]);
        Mu1[Lims1[ii+1,1]:Lims1[ii+1,2]] = - uNUp[ii+1] - uNDown[ii+1] + vec(V[ii+1][3][:,2]);
    end
    ii = nLayer-1;
    Mu1[Lims2[ii,1]:Lims2[ii,2]] = - u1Up[ii+1] - u1Down[ii+1] + vec(V[ii+1][2][:,1]) ;


    Mu = zeros(Complex{Float64},round(Integer,maximum(Lims2)) );


    Mu[Lims1[1,1]:Lims1[1,2]] =  vec(V[1][4][:,4])- uNpDown[1] ;

    for ii=1:nLayer-2
        Mu[Lims2[ii,1]:Lims2[ii,2]] = - u0Up[ii+1] + vec(V[ii+1][1][:,3]);
        Mu[Lims1[ii+1,1]:Lims1[ii+1,2]] = - uNpDown[ii+1] + vec(V[ii+1][4][:,4]);
    end
    ii = nLayer-1;
    Mu[Lims2[ii,1]:Lims2[ii,2]] = -  u0Up[ii+1] + vec(V[ii+1][1][:,3])

    return vcat(Mu1,Mu)

end


function applyDinvDownOptDDM(DDM::DomainDecomposition, uGamma)
    # this function provides with Dinvu and LDinvu simultaneously
    # we use it to encapsulate the application of the Dinv using
    # a class Domain Decomposition
    # obtaining the number of layers
    nLayer = size(DDM.subDomains)[1];
    nSurf = DDM.subDomains[1].model.size[2]*DDM.subDomains[1].model.size[3]
    nInd    = 1:nSurf;

    # for now we will leave the dummyzero intact but we should modify them too
    dummyzero = zeros(Complex{Float64},nSurf);
    Dinvu     = zeros(Complex{Float64},2*(nLayer-1)*nSurf);

    Lims1 = DDM.LimsN;
    Lims2 = DDM.Lims1;

    Dinvu[Lims1[1,1]:Lims1[1,2]] = -uGamma[Lims1[1,1]:Lims1[1,2]];
    Dinvu[Lims2[1,1]:Lims2[1,2]] = -uGamma[Lims2[1,1]:Lims2[1,2]];
    vN   = Dinvu[Lims1[1,1]:Lims1[1,2]] ;
    vNp  = Dinvu[Lims2[1,1]:Lims2[1,2]];

    Lu = zeros(Complex{Float64},round(Integer,maximum(Lims2)) );



    for ii=1:nLayer-2
        uN = uGamma[Lims1[ii+1,1]:Lims1[ii+1,2]];
        uNp= uGamma[Lims2[ii+1,1]:Lims2[ii+1,2]];

        (v0, v1, vNaux, vNpaux) = applyBlockOperator(DDM.subDomains[ii+1],vN,vNp, dummyzero,dummyzero);

        Lu[Lims1[ii,1]:Lims1[ii,2]]  = vec(v0)           ;
        Lu[Lims2[ii,1]:Lims2[ii,2]] = vec(v1) - vec(vNp);

        Dinvu[Lims1[ii+1,1]:Lims1[ii+1,2]]      = vec(vNaux)  - vec(uN);
        Dinvu[Lims2[ii+1,1]:Lims2[ii+1,2]]  = vec(vNpaux) - vec(uNp);

        # update vN and vNp
        vN   = Dinvu[Lims1[ii+1,1]:Lims1[ii+1,2]] ;
        vNp  = Dinvu[Lims2[ii+1,1]:Lims2[ii+1,2]];
    end

    ii = nLayer-1;

    (v0, v1, vNaux, vNpaux) = applyBlockOperator(DDM.subDomains[ii+1],vN,vNp, dummyzero,dummyzero);

    Lu[Lims1[ii,1]:Lims1[ii,2]]  = vec(v0)           ;
    Lu[Lims2[ii,1]:Lims2[ii,2]] = vec(v1) - vec(vNp);

    return (Dinvu, Lu)
end


### TODO
function applyDinvUpOptDDM(DDM::DomainDecomposition, uGamma)
    # function to apply the DupInv but obtaining
    # Uu at the same time for the same cost
    # This would represent pair-wise reflections so it should
    # increase the speed of the convergence
    # obtaining the number of layers
    nLayer = size(DDM.subDomains)[1];
    nSurf = DDM.subDomains[1].model.size[2]*DDM.subDomains[1].model.size[3]
    nInd    = 1:nSurf;

    dummyzero = zeros(Complex{Float64},nSurf);
    Dinvu     = zeros(Complex{Float64},2*(nLayer-1)*nSurf);

    Lims1 = DDM.LimsN;
    Lims2 = DDM.Lims1;

    Uu = zeros(Complex{Float64},2*(nLayer-1)*nSurf);


    # last layer
    ii = nLayer-1
    Dinvu[Lims1[ii,1]:Lims1[ii,2]]  = -uGamma[Lims1[ii,1]:Lims1[ii,2]];
    Dinvu[Lims2[ii,1]:Lims2[ii,2]]  = -uGamma[Lims2[ii,1]:Lims2[ii,2]];
    v0  = Dinvu[Lims1[ii,1]:Lims1[ii,2]];
    v1  = Dinvu[Lims2[ii,1]:Lims2[ii,2]];

    for ii = nLayer-2:-1:1
        u0 = uGamma[Lims1[ii,1]:Lims1[ii,2]];
        u1 = uGamma[Lims2[ii,1]:Lims2[ii,2]];

        (v0aux, v1aux, vN, vNp) = applyBlockOperator(DDM.subDomains[ii+1],dummyzero,dummyzero,v0,v1);

        Dinvu[Lims1[ii,1]:Lims1[ii,2]] = vec(v0aux) - vec(u0);
        Dinvu[Lims2[ii,1]:Lims2[ii,2]] = vec(v1aux) - vec(u1);

        Uu[Lims1[ii+1,1]:Lims1[ii+1,2]] = vec(vN) - vec(v0);
        Uu[Lims2[ii+1,1]:Lims2[ii+1,2]] = vec(vNp) ;

        v0  = Dinvu[Lims1[ii,1]:Lims1[ii,2]] ;
        v1  = Dinvu[Lims2[ii,1]:Lims2[ii,2]];
    end

    ii = 0

    (v0aux, v1aux, vN, vNp) = applyBlockOperator(DDM.subDomains[ii+1],dummyzero,dummyzero,v0,v1);

    Uu[Lims1[ii+1,1]:Lims1[ii+1,2]] = vec(vN) - vec(v0);
    Uu[Lims2[ii+1,1]:Lims2[ii+1,2]] = vec(vNp)  ;


    return (Dinvu, Uu)
end

function extractRHS(DDM::DomainDecomposition,source::Array{Complex128,1})
    #function to produce and extra the rhs

    uLocalArray = solveLocal(DDM.subDomains, source)

    return extractFullBoundaryData(DDM, uLocalArray)
end

# to be redone

function extractFullBoundaryData(DDM::DomainDecomposition, uLocalArray)
    # Function to extract the boundary data from an array of local solutions
    # input   SubArray: subdomain associated to the solution
    #         u        : solution
    # output  (u0, u1, uN, uNp) : tuple of the solution at different depth
    # check size

    # partitioning the source % TODO make it a function
    nSubs = length(DDM.subDomains);

    n = DDM.Lims1[end,2];

    u_0  = zeros(Complex128,round(Integer,n/2))
    u_1  = zeros(Complex128,round(Integer,n/2))
    u_n  = zeros(Complex128,round(Integer,n/2))
    u_np = zeros(Complex128,round(Integer,n/2))

    index = 1:n

    # populate the traces

    for ii = 1:nSubs
        # this will be slow most likely but it is readable
        # we will need to modify this part
        ind_0  = DDM.subDomains[ii].ind_0
        ind_1  = DDM.subDomains[ii].ind_1
        ind_n  = DDM.subDomains[ii].ind_n
        ind_np = DDM.subDomains[ii].ind_np

        if ii != 1
            u_0[DDM.Lims1[ii-1,1]:DDM.Lims1[ii-1,2]] = uLocalArray[ii][ind_0]
            u_1[DDM.Lims1[ii-1,1]:DDM.Lims1[ii-1,2]] = uLocalArray[ii][ind_1]
        end
        if ii !=nSubs
            u_n[DDM.Lims1[ii,1]:DDM.Lims1[ii,2]] = uLocalArray[ii][ind_n]
            u_np[DDM.Lims1[ii,1]:DDM.Lims1[ii,2]] = uLocalArray[ii][ind_np]
        end

    end

    return (u_0, u_1, u_n, u_np)

end

# functions to write

## TODO: Modify this function to be agnostic of the sizes of the local domains
function vectorizePolarizedBdyDataRHS(DDM::DomainDecomposition,uBdyData)
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



#######################################################################################
#                                                                                     #
#                     Preconditioners for the Integral System                         #
#                                                                                     #
#######################################################################################

type IntegralPreconditioner
    n::Int64  # number of grid points in the plane
    nSubs::Int64
    subDomains
    nIt::Int64
    P # permutation matrix
    precondtype::ASCIIString # preconditioner type Jacobi, GaussSeidel Opt
    function IntegralPreconditioner(subDomains; nIt = 1, precondtype ="GS")
        n = subDomains[1].model.size[2]*subDomains[1].model.size[3]
        nSubs = length(subDomains)
        P = generatePermutationMatrix(n,nSubs);
        new(n,nSubs, subDomains,nIt, P, precondtype) # don't know if it's the best answer
    end
end

# Encapsulation of the preconditioner in order to use preconditioned GMRES
import Base.\

function \(M::IntegralPreconditioner, b::Array{Complex128,1})
    #println("Applying the polarized Traces Preconditioner")
    if M.precondtype == "GS"
        return PrecondGaussSeidel(M.subDomains, M.P*b, M.nIt)
    elseif M.precondtype == "GSOpt"
        # using an optimized Gauss-Seidel to have one less solve per iteration
        return PrecondGaussSeidelOpt(M.subDomains, M.P*b, M.nIt)
    elseif M.precondtype == "GSOpt2"
        # using an optimized Gauss-Seidel to have one less solve per iteration
        return PrecondGaussSeidelOpt2(M.subDomains, M.P*b, M.nIt)
    elseif M.precondtype == "Jac"
        return PrecondJacobi(M.subDomains, M.P*b, M.nIt)
    end
end

function PrecondJacobi(subArray, v::Array{Complex128,1}, nit)
    # function to apply the block Jacobi preconditioner using the data
    # TODO add dimension checks
    u = 0*v;
    for ii = 1:nit
        # splitting the vector in two parts
        udown = u[1:round(Integer,end/2)];
        uup   = u[(1+round(Integer,end/2)):end];
        # f - Ru^{n-1}
        if norm(u)!=0
            # skip this step in the first iteration
            udownaux = v[1:round(Integer,end/2)]       - applyU(subArray, uup);
            uupaux   = v[(1+round(Integer,end/2)):end] - applyL(subArray, udown);
        else
            udownaux = v[1:round(Integer,end/2)]
            uupaux   = v[(1+round(Integer,end/2)):end]
        end

        vdown = applyDinvDown(subArray,udownaux);
        vup   = applyDinvUp(subArray,uupaux);
        u    = vcat(vdown, vup );
    end
    return u;
end

function PrecondGaussSeidel(subArray, v::Array{Complex128,1}, nit; verbose=false)
    # function to apply the block Gauss-Seidel Preconditioner
    # input :   subArray  array pointer to the set of subdomains
    #           v         rhs to be solved
    #           nit       number of iterations
    # output:   u         Approximated solution
    # using a first guess equals to zero

    println("Applying the preconditioner");
    u = 0*v;
    for ii = 1:nit

        # splitting the vector in two parts
        udown = u[1:round(Integer,end/2)];
        uup   = u[(1+round(Integer,end/2)):end];

        # f - Ru^{n-1}
        if norm(u)  != 0
            udownaux = v[1:round(Integer,end/2)]       - applyU(subArray, uup);
        else
            udownaux = v[1:round(Integer,end/2)]
        end
        uupaux   = v[(1+round(Integer,end/2)):end] ;

        # applying the inverses
        vdown = applyDinvDown(subArray,udownaux);
        vup   = applyDinvUp(subArray, uupaux - applyL(subArray,vdown));

        if verbose
            println("magnitude of the update = ", norm(u[:] -  vcat(vdown, vup ))/norm(v[:]));
        end

        # concatenatinc the solution
        u    = vcat(vdown, vup );

    end
    return u;
end

function PrecondGaussSeidelOpt(subArray, v::Array{Complex128,1}, nit; verbose=false)
    # function to apply the block Gauss-Seidel Preconditioner
    # in this case we use the jump conditions to eliminate one local
    # solve in order to accelerate the execution time
    # Now we have only 2 local solves per layer per application of the
    # preconditioner
    # input :   subArray  array pointer to the set of subdomains
    #           v         rhs to be solved
    #           nit       number of iterations
    # output:   u         Approximated solution
    # using a first guess equals to zero

    println("Applying the preconditioner");
    u = 0*v;
    for ii = 1:nit

        # splitting the vector in two parts
        udown = u[1:round(Integer,end/2)];
        uup   = u[(1+round(Integer,end/2)):end];

        # f - Ru^{n-1}
        if norm(u)  != 0
            udownaux = v[1:round(Integer,end/2)]       - applyU(subArray, uup);
        else
            udownaux = v[1:round(Integer,end/2)]
        end
        uupaux   = v[(1+round(Integer,end/2)):end] ;

        # applying the inverses
        (vdown, udownaux) = applyDinvDownOpt(subArray,udownaux);
        vup               = applyDinvUp(subArray, uupaux - udownaux);

        if verbose
            println("magnitude of the update = ", norm(u[:] -  vcat(vdown, vup ))/norm(v[:]));
        end

        # concatenating the solution
        u    = vcat(vdown, vup );

    end
    return u;
end

function PrecondGaussSeidelOpt2(subArray, v::Array{Complex128,1}, nit; verbose=false)
    # function to apply the block Gauss-Seidel Preconditioner
    # in this case we use the jump conditions to eliminate one local
    # solve in order to accelerate the execution time
    # Now we have only 2 local solves per layer per application of the
    # preconditioner, in this case we
    # input :   subArray  array pointer to the set of subdomains
    #           v         rhs to be solved
    #           nit       number of iterations
    # output:   u         Approximated solution
    # using a first guess equals to zero

    println("Applying the preconditioner Gauss-Seidel Optimized 2");
    u = 0*v;
    # the for loops are not leaky then we need to define the wavefields outisde
    vdown = 0*v;
    udownaux = 0*v;
    vup = 0*v ;
    for ii = 1:nit

        # splitting the vector in two parts
        udown = u[1:round(Integer,end/2)];
        uup   = u[(1+round(Integer,end/2)):end];

        # f - Ru^{n-1}
        if norm(u)  != 0
            udownaux = v[1:round(Integer,end/2)]  - udownaux;
        else
            udownaux = v[1:round(Integer,end/2)]
        end
        uupaux   = v[(1+round(Integer,end/2)):end] ;

        # applying the inverses
        (vdown, udownaux) = applyDinvDownOpt(subArray,udownaux);
        (vup  , udownaux) = applyDinvUpOpt(subArray, uupaux - udownaux);


        if verbose
            println("magnitude of the update = ", norm(u[:] -  vcat(vdown, vup ))/norm(v[:]));
        end

        # concatenating the solution
        u  = vcat(vdown, vup );

    end
    u = vcat(vdown + udownaux, vup );
    return u;
end

#######################################################################################
#                                                                                     #
#        Preconditioners for the Integral System using the DMM encapsulation          #
#                                                                                     #
#######################################################################################

type IntegralPreconditionerDDM
    nSubs::Int64
    DDM::DomainDecomposition
    nIt::Int64
    P # permutation matrix
    precondtype::ASCIIString # preconditioner type Jacobi, GaussSeidel Opt
    function IntegralPreconditionerDDM(DDM::DomainDecomposition; nIt = 1, precondtype ="GSOpt2")
        nSubs = length(DDM.subDomains)
        # TODO this needs to be modified in order to allow for different
        n = DDM.LimsN[1,2];
        # we need to fix the permutation matrix, this is going to be completely different if the
        # sizes of the interface data are different
        P = generatePermutationMatrix(n,nSubs);
        new(nSubs, DDM,nIt, P, precondtype) # don't know if it's the best answer
    end
end

# Encapsulation of the preconditioner in order to use preconditioned GMRES
import Base.\

function \(M::IntegralPreconditionerDDM, b::Array{Complex128,1})
    # #println("Applying the polarized Traces Preconditioner")
    # if M.precondtype == "GS"
    #     return PrecondGaussSeidel(M.DDM, M.P*b, M.nIt)
    # elseif M.precondtype == "GSOpt"
    #     # using an optimized Gauss-Seidel to have one less solve per iteration
    #     return PrecondGaussSeidelOpt(M.DDM, M.P*b, M.nIt)
    # elseif M.precondtype == "GSOpt2"
        # using an optimized Gauss-Seidel to have one less solve per iteration

        return PrecondGaussSeidelOpt2DDM(M.DDM, M.P*b, M.nIt)
    # elseif M.precondtype == "Jac"
    #     return PrecondJacobi(M.DDM, M.P*b, M.nIt)
    # end
end


function PrecondGaussSeidelOpt2DDM(DDM::DomainDecomposition, v::Array{Complex128,1}, nit; verbose=false)
    # function to apply the block Gauss-Seidel Preconditioner
    # in this case we use the jump conditions to eliminate one local
    # solve in order to accelerate the execution time
    # Now we have only 2 local solves per layer per application of the
    # preconditioner, in this case we
    # input :   DDM       encapsulation class for the domain decomposition
    #           v         rhs to be solved
    #           nit       number of iterations
    # output:   u         Approximated solution
    # using a first guess equals to zero

    println("Applying the preconditioner Gauss-Seidel Optimized 2");
    u = 0*v;
    vdown = 0*v;
    udownaux = 0*v;
    vup = 0*v ;

    for ii = 1:nit

        # splitting the vector in two parts
        udown = u[1:round(Integer,end/2)];
        uup   = u[(1+round(Integer,end/2)):end];

        # f - Ru^{n-1}
        if norm(u)  != 0
            udownaux = v[1:round(Integer,end/2)]  - udownaux;
        else
            udownaux = v[1:round(Integer,end/2)]
        end
        uupaux   = v[(1+round(Integer,end/2)):end] ;

        # applying the inverses
        (vdown, udownaux) = applyDinvDownOptDDM(DDM,udownaux);
        (vup  , udownaux) = applyDinvUpOptDDM(DDM, uupaux - udownaux);


        if verbose
            println("magnitude of the update = ", norm(u[:] -  vcat(vdown, vup ))/norm(v[:]));
        end

        # concatenating the solution
        u  = vcat(vdown, vup );

    end
    u = vcat(vdown + udownaux, vup );
    return u;
end

type PolarizedTracesPreconditioner
    subArray
    P   # permutation matrix
    nIt::Int64
    typePrecond::ASCIIString

     function PolarizedTracesPreconditioner(subArray, P; typePrecond="GS",nIt = 1)
        new(subArray,P, nIt, typePrecond) # don't know if it's the best answer
    end
end



# Encapsulation of the preconditioner in order to use preconditioned GMRES
import Base.\

function \(M::PolarizedTracesPreconditioner, v::Array{Complex128,1})
    #println("Applying the polarized Traces Preconditioner")
    # Allocating the space
    u = zeros(v)
    if M.typePrecond == "GS"
        u =  PrecondGaussSeidel(M.subArray, M.P*v, M.nIt)
    elseif M.typePrecond == "Jac"
        u =  PrecondJacobi(M.subArray, M.P*v, M.nIt)
    end

    return u
end


function PrecondJacobi(subArray, v::Array{Complex128,1}, nit::Int64)
    # function to apply the block Gauss-Seidel Preconditioner
    # input :   subArray  array pointer to the set of subdomains
    #           v         rhs to be solved
    #           nit       number of iterations
    # output:   u         Approximated solution
    # using a first guess equals to zero
    u = 0*v;
    for ii = 1:nit
        # splitting the vector in two parts
        udown = u[1:end/2];
        uup   = u[(1+end/2):end];
        # f - Ru^{n-1}
        udownaux = v[1:end/2]       - applyU(subArray, uup);
        uupaux   = v[(1+end/2):end] - applyL(subArray, udown);

        vdown = applyDinvDown(subArray,udownaux);
        vup   = applyDinvUp(subArray,uupaux);
        u    = vcat(vdown, vup );
    end
    return u;
end


function PrecondGaussSeidel(subArray, v::Array{Complex128,1}, nit::Int64)
    # function to apply the block Gauss-Seidel Preconditioner
    # input :   subArray  array pointer to the set of subdomains
    #           v         rhs to be solved
    #           nit       number of iterations
    # output:   u         Approximated solution
    # using a first guess equals to zero
    u = 0*v;
    for ii = 1:nit

        # splitting the vector in two parts
        udown = u[1:round(Int,end/2)];
        uup   = u[(1+round(Int,end/2)):end];

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

        # concatenatinc the solution
        u    = vcat(vdown, vup );

    end
    return u;
end

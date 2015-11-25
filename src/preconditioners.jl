function PrecondJacobi(subArray, v, nit)
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


function PrecondGaussSeidel(subArray, v, nit)
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
        uupaux   = v[(1+end/2):end] ;

        # applying the inverses
        vdown = applyDinvDown(subArray,udownaux);
        vup   = applyDinvUp(subArray, uupaux - applyL(subArray,vdown));

        # concatenatinc the solution
        u    = vcat(vdown, vup );

    end
    return u;
end

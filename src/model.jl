# class model
# in this type we save all the information related to the physics and discretization at
# each layer. In this case we use a simple second order finite difference discretization

include("HelmholtzMatrix.jl");


type Model
    H::SparseMatrixCSC{Complex{Float64},Int64} # sparse matrix
    # local meshes
    h::Float64
    x     #mesh in x
    y
    z
    zExt
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
    solvertype
    function Model(m::Array{Float64,3},npml::Int64,zExt, bdy,h::Float64,fac::Float64,order::Int64,omega::Float64; solvertype = "UMFPACK")
        # m = matrix(nx,ny,nz)
        # extracting the size of the 3D domain
        (nx,ny,nz) = size(m);
        # building the differential operator
        #TO DO change how the system is built to have pml of different sizes in
        # every dimension
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

        new(H,h,x,y,z,zExt,xLim,yLim,zLim,[7 3],[8 2],zLimInd,[], [nx*ny*nz,nx,ny,nz], solvertype) # don't know if it's the best answer
    end
end


 # m = matrix(nx,ny,nz)
function extractBoundaryIndices(localModel::Model);
  	return (localModel.zLimInd[1,:], localModel.zLimInd[2,:], localModel.zLimInd[3,:], localModel.zLimInd[4,:])
end

# for the following matrices we need more informaiton
function extractVolIntIndices(localModel::Model);
 	zInd1 = find(abs(localModel.zExt-localModel.zLim[2]).< localModel.h/10)[1]
 	zIndn = find(abs(localModel.zExt-localModel.zLim[3]).< localModel.h/10)[1]

 	return collect((localModel.size[3]*localModel.size[2]*(zInd1-1)+1):(localModel.size[3]*localModel.size[2]*zIndn))
end

function extractVolIndices(localModel::Model);
 	zInd1 = find(abs(localModel.zExt-localModel.zLim[2]).< localModel.h/10)[1]
 	zIndn = find(abs(localModel.zExt-localModel.zLim[3]).< localModel.h/10)[1]

 	return collect((localModel.size[3]*localModel.size[2]*(zInd1-1)+1):(localModel.size[3]*localModel.size[2]*zIndn))

end

function extractVolIntLocalIndices(localModel::Model);
	zInd1 = find(abs(localModel.z - localModel.zLim[2]).< localModel.h/10)[1]
 	zIndn = find(abs(localModel.z - localModel.zLim[3]).< localModel.h/10)[1]

 	return collect((localModel.size[3]*localModel.size[2]*(zInd1-1)+1):(localModel.size[3]*localModel.size[2]*zIndn))
end


function factorize!(model::Model)
        # TODO : factorization has to be performed separately
        println("Factorizing the local matrix")
        if model.solvertype == "UMFPACK"
            model.Hinv = lufact(model.H);
        end

        if model.solvertype == "MKLPARDISO"
            model.Hinv = MKLPardisoSolver();
            set_nprocs(model.Hinv, 16)
            #setting the type of the matrix
            set_mtype(model.Hinv,3)
            # setting we are using a transpose
            set_iparm(model.Hinv,12,2)
            # setting the factoriation phase
            set_phase(model.Hinv, 12)
            X = zeros(Complex128, model.size[1],1)
            # factorizing the matrix
            pardiso(model.Hinv,X, model.H,X)
            # setting phase and parameters to solve and transposing the matrix
            # this needs to be done given the different C and Fortran convention
            # used by Pardiso (C convention) and Julia (Fortran Convention)
            set_phase(model.Hinv, 33)
            set_iparm(model.Hinv,12,2)
        end
end



function convert64_32!(model::Model)
    if model.solvertype == "MKLPARDISO"
        model.H = SparseMatrixCSC{Complex128,Int32}(model.H)
    else
        println("This method is only to make PARDISO more efficient")
    end
end

function solve(model::Model, f::Array{Complex128,1})
    # u = solve(model::Model, f)
    # function that solves the system Hu=f in the model for one RHS
    # check size
    if (size(f[:])[1] == model.size[1])
        if model.solvertype == "UMFPACK"
            u = model.Hinv\f[:];
        end
        # if the linear solvers is MKL Pardiso
        if model.solvertype == "MKLPARDISO"
            set_phase(model.Hinv, 33)
            u = zeros(Complex128,length(f))
            pardiso(model.Hinv, u, model.H, f)
        end

        return u
    else
        print("The dimensions do not match \n");
        return 0
    end
end

function solve(model::Model, f::Array{Complex128,2})
    # u = solve(model::Model, f)
    # function that solves the system Hu=f in the model
    # for multiple RHS simultaneously
    # check size
    if (size(f)[1] == model.size[1])
        if model.solvertype == "UMFPACK"
            u = model.Hinv\f;
        end
        # if the linear solvers is MKL Pardiso
        if model.solvertype == "MKLPARDISO"
            set_phase(model.Hinv, 33)
            u = zeros(f)
            pardiso(model.Hinv, u, model.H, f)
        end

        return u
    else
        print("The dimensions do not match \n");
        return 0
    end
end


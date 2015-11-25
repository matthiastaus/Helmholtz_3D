# File with the functions to create finite differences stencils
# plus PML. This file is the basis for the construction of the
# Helmholtz operator

function FDweights(z,x,m::Int)
#---------------------------------
# finite-difference weights
# (Fornberg algorithm)
#
# z:  expansion point
# x:  vector of evaluation points
# m:  order of derivative
#
# Example: cwei = FDweights(0,[0 1 2],1);
# gives    cwei = [-3/2  2  -1/2]
#
# h f'_0 = -3/2 f_0 + 2 f_1 - 1/2 f_2
#
#---------------------------------

  n  = length(x)-1;
  c1 = 1;
  c4 = x[1]-z;
  c = zeros(n+1,m+1);
  c[1,1] = 1;
  for i=1:n
    mn = min(i,m);
    c2 = 1;
    c5 = c4;
    c4 = x[i+1]-z;
    for j=0:i-1
      c3 = x[i+1]-x[j+1];
      c2 = c2*c3;
      if (j == (i-1))
        for k=mn:-1:1
          c[i+1,k+1] = c1*(k*c[i,k]-c5*c[i,k+1])/c2;
        end
        c[i+1,1] = -c1*c5*c[i,1]/c2;
      end
      for k=mn:-1:1
        c[j+1,k+1] = (c4*c[j+1,k+1]-k*c[j+1,k])/c3;
      end
      c[j+1,1] = c4*c[j+1,1]/c3;
    end
    c1 = c2;
  end
  cwei = c[:,end];
  return cwei
end

function FirstOrderDifferenceMatrix1d(nx::Int,h::FloatingPoint,order::Int)
  # function Dx = FirstOrderDifferenceMatrix1d(nx,h::FloatingPoint,order::Int)
  # function to compute the first order finite different matrix using
  # the Fornberg algorithm to compute the stencil, with a descentered stencil
  # at the edges
  # input :   nx    size of the matrix to be generated
  #           h     discretization step used
  #           order order of the discretization used
  # output:   Dx    Finite difference Matrix

  #computing the FD matrix at the interior
  diagonals = repmat(FDweights(order/2,linspace(0,order,order+1),1)'/h,nx,1);
  Dx = spzeros(nx,nx);
  for ii = 1:order+1
    bound = abs(-int(order/2)-1+ii);
    Dx = Dx + spdiagm(diagonals[1:end-int(bound),ii] ,-int(order/2)-1+ii,nx,nx);
  end

  # modifing the matrix at the boundaries, using descentered stencils
   for ii = 1:(int(order/2)-1)
     weights = FDweights(ii,linspace(0,order+2,order+3),1)'/h;
     Dx[ii,1:order+2]=weights[2:end];

     weights = FDweights(order+2-ii,linspace(0,order+2,order+3),1)'/h;
     Dx[end-(ii-1),(end-(order+1)):end]=weights[1:end-1];
   end
  return Dx
end

function stiffness_matrix(nx::Int, dx::FloatingPoint, order::Int)
  # function Dxx = stiffness_matrix(nx::Int, dx::FloatingPoint, order::Int)
  # function to compute a 1D stiffness Matrix using  finite differences
  # (Dirichlet boundary nodes are not on the grid)
  # input :   nx    size of the matrix to be generated
  #           dx    discretization step used
  #           order order of the discretization used
  # output:   Dxx   Stiffness Matrix

  # computing the FD matrix at the interior
  diagonals = repmat(FDweights(order/2,linspace(0,order,order+1),2)'/(dx^2),nx,1);
  Dxx = spzeros(nx,nx);

  # creating the matrix Dxx, this should be modified but spdiagm only accepts one
  # diagonal at a time
  for ii = 1:order+1
    bound = abs(-int(order/2)-1+ii);
    Dxx = Dxx + spdiagm(diagonals[1:end-int(bound),ii],-int(order/2)-1+ii,nx,nx);
  end

  # modifying the matrix at the boundaries to obtain an uniform accuracy
  # using descentered stencil at the boundaries
  for ii = 1:(int(order/2)-1)
    # modifying on end of the matrix with the correct descentered stencil
     weights = FDweights(ii,linspace(0,order+2,order+3),2)'/(dx^2);
     Dxx[ii,1:order+2]=weights[2:end];

     # modifying the other end of the matrix with the correct descentered stencil
     weights = FDweights(order+2-ii,linspace(0,order+2,order+3),2)'/(dx^2);
     Dxx[end-(ii-1),(end-(order+1)):end]=weights[1:end-1];
   end

   return Dxx
end

function DistribPML(nx::Int,ny::Int,nz::Int, nPML::Int,fac::FloatingPoint)
  # function (sigmaX, sigmaY, sigmaZ) = DistribPML(nx,ny,nz, nPML,fac)
  # function to create the damping profile on the PML's
  # input :   nx   number of poinst in the x direction
  #           ny   number of poinst in the y direction
  #           nz   number of poinst in the z direction
  #           nPML number of PML points at each side
  #           fact absorption factor for PML
  # output:   (sigmaX, sigmaY, sigmaZ) pml profile in each direction

  # this is a compressed for loop, I don't know if it's vectorized or not
  sigmaX = [fac*sigma(i,nx,nPML)+0*j+0*k for i=1:nx,j=1:ny,k=1:nz ];
  sigmaY = [fac*sigma(j,ny,nPML)+0*i+0*k for i=1:nx,j=1:ny,k=1:nz ];
  sigmaZ = [fac*sigma(k,nz,nPML)+0*i+0*j for i=1:nx,j=1:ny,k=1:nz ];

  return (sigmaX, sigmaY, sigmaZ)
end

function DistribPMLDerivative(nx::Int,ny::Int,nz::Int, nPML::Int,fac::FloatingPoint)
  # function to create the derivative of the damping profile on the PML's
  # this is a compressed for loop, i don't know if it's vectorized or not

  DxsigmaX = [fac*Dxsigma(i,nx,nPML)+0*j+0*k for i=1:nx,j=1:ny,k=1:nz ];
  DysigmaY = [fac*Dxsigma(j,ny,nPML)+0*i+0*k for i=1:nx,j=1:ny,k=1:nz ];
  DzsigmaZ = [fac*Dxsigma(k,nz,nPML)+0*i+0*j for i=1:nx,j=1:ny,k=1:nz ];

  return (DxsigmaX, DysigmaY, DzsigmaZ)
end


function sigma(i::Int,n::Int,nPML::Int)
  # pml function, we start one point after the boundary to enforce the continuity
  res = (i.<nPML).*((i-nPML).^2)/(nPML-1).^2 + (i.> (n-nPML+1)).*((i-n+nPML-1).^2)/(nPML-1).^2;
end

function Dxsigma(i::Int,n::Int,nPML::Int)
  # derivative of the pml function, we start one point after the boundary to enforce the continuity
  res = -2*(i.<nPML).*((i-nPML))/(nPML-1).^2 + 2*(i.> (n-nPML+1)).*((i-n+nPML-1))/(nPML-1).^2;
end

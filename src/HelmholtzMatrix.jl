include("FDweights.jl")

function HelmholtzMatrix(m,nx::Int64,ny::Int64,nz::Int64,npml::Int64,h::Float64,
                         fac::Float64,order::Int64,omega; profileType="quadratic")
  #function HelmholtzMatrix(m,nx,ny,nz,npml,h,fac,order,omega)

  #  H = -(\triangle + \omega^2 m I)
  # total number of degrees of freedom
  n = nx*ny*nz;
  if profileType == "quadratic"
    (sx,sy,sz)    = DistribPML(nx,ny,nz,npml,fac);
    (dsx,dsy,dsz) = DistribPMLDerivative(nx,ny,nz,npml,fac);
    (dsx,dsy,dsz) = (dsx/((npml-1)*h),dsy/((npml-1)*h),dsz/((npml-1)*h))  

  elseif  profileType == "unbounded"
    c = 1./sqrt(m)
    (sx,sy,sz)    = DistribPML(nx,ny,nz,npml,fac,
                               c=c,profileType=profileType);
    (dsx,dsy,dsz) = DistribPMLDerivative(nx,ny,nz,npml,fac,
                                         c=c,profileType=profileType);
    # we need to rescale the profiles 
    (sx,sy,sz) = (sx/h,sy/h,sz/h);
    (dsx,dsy,dsz) =  (dsx./h^2,dsy./h^2,dsz./h^2);

  end

  # assembling the 1-dimensional stiffness matrices
  Dxx1d = stiffness_matrix(nx,h,order);
  Dyy1d = stiffness_matrix(ny,h,order);
  Dzz1d = stiffness_matrix(nz,h,order);
  # assembking the 1-dimensional finite difference matrices
  Dx1d  = FirstOrderDifferenceMatrix1d(nx,h,order);
  Dy1d  = FirstOrderDifferenceMatrix1d(ny,h,order);
  Dz1d  = FirstOrderDifferenceMatrix1d(nz,h,order);

  # assembling the 3D matrices using Kronecker deltas
  Dx    = kron(kron(speye(nz),speye(ny)),Dx1d    );
  Dy    = kron(kron(speye(nz),Dy1d)    ,speye(nx));
  Dz    = kron(kron(Dz1d    ,speye(ny)),speye(nx));

  Dxx   = kron(kron(speye(nz),speye(ny)),Dxx1d    );
  Dyy   = kron(kron(speye(nz),Dyy1d)    ,speye(nx));
  Dzz   = kron(kron(Dzz1d    ,speye(ny)),speye(nx));

  # assembling the slowness matrix
  M     = spdiagm(m[:],0,n,n);

# re do the matrix for Helmholtz equation in 3D
  H = - omega^2*M +
        spdiagm(-1.im/omega*dsx[:]./(1-1.im/omega*sx[:]).^3,0,n,n)*Dx +
        spdiagm(-1.im/omega*dsy[:]./(1-1.im/omega*sy[:]).^3,0,n,n)*Dy +
        spdiagm(-1.im/omega*dsz[:]./(1-1.im/omega*sz[:]).^3,0,n,n)*Dz -
        spdiagm(1./(1-1.im/omega*sx[:]).^2,0,n,n)*Dxx-
        spdiagm(1./(1-1.im/omega*sy[:]).^2,0,n,n)*Dyy-
        spdiagm(1./(1-1.im/omega*sz[:]).^2,0,n,n)*Dzz;
  return H;



end

using FFTW
using PlotlyJS
using CUDA



# xx & yy for plotting [only]
# [xxp,yyp] = meshgrid(x,y) - creates a copy of rows of x and columns of y
# julia equivalent
function meshgrid(x, y)
    nx, ny = length(x), length(y)
    xxp = ones(nx, nx) .* x'
    yyp = ones(ny, ny) .* y
    return [xxp, yyp]
end


# Tested functions

function omg2vel(omghat)
    # %% Compute (u,v,omega,psi) from omega hat 
    # % input in wave space
    # % output in real space
    
    uhat = zeros(size(omghat)) + 1im * zeros(size(omghat)); #forcing uhat to be complex 
    vhat = copy(uhat);
    psihat = copy(uhat); 
    omg = real(ifft(omghat));
    
    for i = 1:nx 
        for j = 1:ny
            if (i*j)==1 
                psihat[j,i] = 0;
            else
                psihat[j,i] = omghat[j,i] / (kx[i]^2 + ky[j]^2);
            end
        end
    end
    
    for i = 1:nx 
        for j = 1:ny
            uhat[j,i] =  1im * ky[j] * psihat[j,i];
            vhat[j,i] = -1im * kx[i] * psihat[j,i];
        end
    end
    
    psi = real(ifft(psihat));
    u   = real(ifft(uhat));
    v   = real(ifft(vhat));
    
    return [u,v,omg,psi]
end

function taylorvtx(x₀, y₀, a₀, U_max)
    omg = zeros(size(xx))
    r2 = 0
    for i = -1:1
        for j = -1:1
            r2 = (xx .- x₀ .- i*Lx).^2 + (yy .- y₀ .- j*Ly).^2;
            # print(r2)
            omg = omg + U_max/a₀*(2 .- r2 / a₀^2).*exp.(0.5*(1 .- r2 / a₀^2));
            # omg Tested
        end
    end
    # print(size(omg))
    omghat = fft(omg)
    # omghat: inaccurate in imaginary space
    return omghat
end


function ic(caseno, rng = nothing) 
    # caseno = 1;
    
    if caseno == 1
        omghat = taylorvtx(Lx/2,Ly/2,Lx/8,1.0);
    
    elseif caseno == 2
        # % case = 2
        # % -- two co-rotating Taylor vortices
        omghat =          taylorvtx(Lx/2,0.4*Ly,Lx/10.0,1.0);
        # print(omghat)
        omghat = omghat + taylorvtx(Lx/2,0.6*Ly,Lx/10,1.0);
    elseif caseno == 3
        # % case = 3
        # % -- multiple random Taylor vortices
        nv = 30;
        if isnothing(rng)
            omghat = taylorvtx(first(rand(1)*Lx), first(rand(1)*Ly), Lx/20, first(rand(1)*2-[1.0]));
        else
            omghat = taylorvtx(first(rand(rng, 1)*Lx), first(rand(rng, 1)*Ly), Lx/20, first(rand(rng, 1)*2-[1.0]));
        end

        for i = 2:nv
            if isnothing(rng)
                omghat = omghat + taylorvtx(first(rand(1)*Lx),first(rand(1)*Ly),Lx/20,first(rand(1)*2-[1.0]));
            else
                omghat = omghat + taylorvtx(first(rand(rng, 1)*Lx),first(rand(rng, 1)*Ly),Lx/20,first(rand(rng, 1)*2-[1.0]));
            end
        end
    elseif caseno == 4
        # % case = 4
        # % -- even more multiple random Taylor vortices
        nv = 50;
        if isnothing(rng)
            omghat = taylorvtx(first(rand(1)*Lx), first(rand(1)*Ly), Lx/20 * (0.5+rand()), first(rand(1)*2-[1.0]));
        else
            omghat = taylorvtx(first(rand(rng, 1)*Lx), first(rand(rng, 1)*Ly), Lx/20 * (0.5+rand(rng)), first(rand(rng, 1)*2-[1.0]));
        end

        for i = 2:nv
            if isnothing(rng)
                omghat = omghat + taylorvtx(first(rand(1)*Lx),first(rand(1)*Ly), Lx/20 * (0.5+rand()) ,first(rand(1)*2-[1.0]));
            else
                omghat = omghat + taylorvtx(first(rand(rng, 1)*Lx),first(rand(rng, 1)*Ly), Lx/20 * (0.5+rand(rng)) ,first(rand(rng, 1)*2-[1.0]));
            end
        end
    end
    return omghat
end

function rk4(f,p,dt)
    # %% Fourth order Runge-Kutta method
    
    k1 = rhs(f, p);
    k2 = rhs(f+0.5*dt*k1, p);
    k3 = rhs(f+0.5*dt*k2, p);
    k4 = rhs(f+dt*k3, p);
    
    fnew = f + dt/6*(k1+2*(k2+k3)+k4);
    return fnew
end

function rhs(omghat, p)
    # %% RHS (of vorticity transport equation) calculator
    # global kx ky nx ny nu

    lin = -nu * (kx2ky2 .* omghat)
    
    # % nonlinear advection added
    rhs = lin + advection(omghat) + p;
    return rhs
end

function advection(omghat)
    # %% Computes the nonlinear advection term with/without de-aliasing
    
    # -------------------------------------------------------------------------------
    # can call omg2vel here
    # % solve for stream function first (no-padding needed)

    psihat  = omghat ./ kx2ky2
    CUDA.@allowscalar psihat[1,1] = 0.0

    # % "unpadded" d(omega)/dx and d(omega)/dy for advection term in wave space
    domgdx = 1im * omghat .* kx_repeat
    domgdy = 1im * omghat .* ky_repeat
# ---------------------------------------------------------------------------------------
    # % "unpadded" u and v in wave space
    vhat = -1im * psihat .* kx_repeat
    uhat = 1im * psihat .* ky_repeat
    
    if ifpad==1
        # % compute advection term with padding
        # % NOTE: kxp = [0:(nxp/2),(-nxp/2+1):(-1)]/Lx*2*pi;
        # %       kyp = [0:(nyp/2),(-nyp/2+1):(-1)]/Ly*2*pi;
        
        # % compute u, v, d(omega)/dx, d(omega)/dy with padding in real space
        up = real(ifft(pad(uhat)));     
        vp = real(ifft(pad(vhat)));
        domgdxp = real(ifft(pad(domgdx))); 
        domgdyp = real(ifft(pad(domgdy)));
        
        # % output in wavespace and chop higher freq components 
        temp = fft( -up.*domgdxp - vp.*domgdyp )
        nonlin = chop(temp)*1.5*1.5;
    else
        # % compute advection term without padding
        # % NOTE: kx  = [0:(nx/2),(-nx/2+1):(-1)]/Lx*2*pi;
        # %       ky  = [0:(ny/2),(-ny/2+1):(-1)]/Ly*2*pi;
    
        # % unpadded u and v in real space
        u = real(ifft(uhat));
        v = real(ifft(vhat));
        
        # % nonlinear advection in unpadded real space
        nonlin = fft(-u.*real(ifft(domgdx)) - v.*real(ifft(domgdy)));
    end
    return nonlin
end

function pad(f);
    # %% Padding in wavespace (padding to store spurious high freq components)
    # % NOTE: kxp = [0:(nxp/2),(-nxp/2+1):(-1)]/Lx*2*pi;
    # %       kyp = [0:(nyp/2),(-nyp/2+1):(-1)]/Ly*2*pi;
    
    # global nx ny nxp nyp
    
    fp = zeros(Int(nyp),Int(nxp)) + 1im*zeros(Int(nyp),Int(nxp));
    fp = send_to_device(device(f), fp)
    ny_half = Int(ny/2);
    nx_half = Int(nx/2);

    
    fp[1:ny_half+1,1:nx_half+1]       = f[1:ny_half+1,1:nx_half+1];
    fp[1:ny_half+1,end-nx_half+2:end] = f[1:ny_half+1,nx_half+2:end];
    fp[end-ny_half+2:end,1:nx_half+1] = f[ny_half+2:end,1:nx_half+1];
    fp[end-ny_half+2:end,end-nx_half+2:end] = f[ny_half+2:end,nx_half+2:end];
    return fp
end

function chop(fp)
    # %% Chopping in wavespace (remove spurious high frequency components)
    # global nx ny 
    # % chopping in wave space
    # % NOTE: kxp = [0:(nxp/2),(-nxp/2+1):(-1)]/Lx*2*pi;
    # %       kyp = [0:(nyp/2),(-nyp/2+1):(-1)]/Ly*2*pi;
    
    f = zeros(ny,nx) + 1im*zeros(ny,nx);
    f = send_to_device(device(fp), f)
    ny_half = Int(ny/2);
    nx_half = Int(nx/2);
    
    f[1:ny_half+1,1:nx_half+1] = fp[1:ny_half+1,1:nx_half+1];
    f[1:ny_half+1,nx_half+2:end] = fp[1:ny_half+1,end-nx_half+2:end];
    f[ny_half+2:end,1:nx_half+1] = fp[end-ny_half+2:end,1:nx_half+1];
    f[ny_half+2:end,nx_half+2:end] = fp[end-ny_half+2:end,end-nx_half+2:end];
    return f
end

function f2fplot(f)
    # %% Output function over [0,Lx]x[0,Ly]
    # % Note that the code solve for f which does not include x=Lx and y=Ly.
    
    fplot = [f[1:end, 1:end] f[1:end, 1]];
   
    fplot = [fplot[1:end,1:end]
             fplot[1,1:end]']
    return fplot
end


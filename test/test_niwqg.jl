function test_set_q()
  prob = NIWQG.Problem()
  q = rand(prob.grid.nx, prob.grid.ny)
  qh = rfft(q)
  NIWQG.set_q!(prob, q)
  isapprox(q, prob.vars.q) && isapprox(qh, prob.sol[1])
end

function test_set_phi()
  prob = NIWQG.Problem()
  phi = rand(prob.grid.nx, prob.grid.ny) + im*rand(prob.grid.nx, prob.grid.ny)
  phih = fft(phi)
  NIWQG.set_phi!(prob, phi)
  isapprox(phi, prob.vars.phi) && isapprox(phih, prob.sol[2])
end

"Test nonlinear PV advection terms in NIWQG by measuring the propoagation speed of the Lamb dipole."
function test_niwqg_lambdipole(; nx=256, dt=1e-3, Lx=2π, Ue=1, Re=Lx/20, ti=Lx/Ue*0.01, nm=3,
                              stepper="RK4")
  nt = round(Int, ti/dt)
  prob = NIWQG.Problem(nx=nx, Lx=Lx, dt=dt, stepper=stepper)
  q0 = lambdipole(Ue, Re, prob.grid)
  NIWQG.set_q!(prob, q0)

  xq = zeros(nm)   # centroid of abs(q)
  Ue_m = zeros(nm) # measured dipole speed
  x, y = gridpoints(prob.grid)
  q = prob.vars.q

  for i = 1:nm # step forward
    stepforward!(prob, nt)
    NIWQG.updatevars!(prob)
    xq[i] = mean(abs.(q).*x) / mean(abs.(q))
    if i > 1
      Ue_m[i] = (xq[i]-xq[i-1]) / ((nt-1)*dt)
    end
  end
  isapprox(Ue, mean(Ue_m[2:end]), rtol=rtol_lambdipole)
end

"Test the group velocity of waves in NIWQG. This is essentially a test of the phi-dispersion term."
function test_niwqg_groupvelocity(; nkw=16, nx=128, Lx=2π, f=1, eta=1/64, uw=1e-2, rtol=1e-3, del=Lx/10,
                                  stepper="RK4")
  kw = nkw*2π/Lx
   σ = eta*kw^2/2
  tσ = 2π/(f+σ)
  dt = tσ/100
  nt = round(Int, 3tσ/(2dt)) # 3/2 a wave period
  cga = eta*kw # analytical group velocity

  prob = NIWQG.Problem(nx=nx, Lx=Lx, dt=dt, f=f, eta=eta, stepper=stepper)
  envelope(x, y) = exp(-x^2/(2*del^2))
  NIWQG.set_planewave!(prob, uw, nkw; envelope=envelope)

  t₋₁ = prob.t
  xw₋₁, yw₋₁ = wavecentroid_niwqg(prob)

  stepforward!(prob, nt)
  NIWQG.updatevars!(prob)
  xw, yw = wavecentroid_niwqg(prob)
  cgn = (xw-xw₋₁) / (prob.t-t₋₁)

  isapprox(cga, cgn, rtol=rtol)
end

"Test q forcing."
function test_niwqg_forcing_q(; dt=0.01, stepper="RK4", nsteps=1, nx=32, Lx=2π, eta=1, kap=0.01)
  k = 4π/Lx # wavenumber 2
  g = TwoDGrid(nx, Lx)
  x, y = gridpoints(g)
  tf = nsteps*dt

  q0 = @. sin(k*x)
  qforcing = @. kap*q0
  qforcingh = rfft(qforcing)

  function calcF!(Fc, Fr, t, s, v, p, g)
    @. Fr = qforcingh
    nothing
  end

  prob = NIWQG.Problem(nx=nx, Lx=Lx, dt=dt, eta=eta, calcF=calcF!, kap=kap, nkap=0, stepper=stepper)
  NIWQG.set_q!(prob, q0)

  stepforward!(prob, nsteps)
  NIWQG.updatevars!(prob)

  isapprox(prob.vars.q, q0, rtol=rtol_niwqg)
end

"Test phi diffusion, hyperdiffusion, dispersion and forcing."
function test_niwqg_forcing_phi(; dt=0.01, stepper="RK4", nsteps=1, nx=32, Lx=2π, eta=0.0, nu=0.01, nnu=0,
                               muw=0.02, nmuw=1)
  k = 4π/Lx # wavenumber 2
  g = TwoDGrid(nx, Lx)
  x, y = gridpoints(g)
  tf = nsteps*dt

  phi0 = @. exp(im*k*x) # nonlinear terms are 0
  phiforcing = @. nu*k^(2nnu)*phi0 + muw*k^(2nmuw)*phi0 + 0.5*im*eta*k^2*phi0
  phiforcingh = fft(phiforcing)

  function calcF!(Fc, Fr, t, s, v, p, g)
    @. Fc = phiforcingh
    nothing
  end

  prob = NIWQG.Problem(nx=nx, Lx=Lx, dt=dt, eta=eta, nu=nu, nnu=nnu, muw=muw, nmuw=nmuw, 
                       calcF=calcF!, stepper=stepper)
  NIWQG.set_phi!(prob, phi0)

  stepforward!(prob, nsteps)
  NIWQG.updatevars!(prob)

  isapprox(prob.vars.phi, phi0, rtol=rtol_niwqg)
end

"Test the refraction term in the phi-equation."
function test_niwqg_nonlinear1(; dt=0.01, stepper="RK4", nsteps=100, nx=32, Lx=2π, kap=0.01)
                               
  k = 4π/Lx # wavenumber 2
  g = TwoDGrid(nx, Lx)
  x, y = gridpoints(g)
  tf = nsteps*dt

  phi0 = fill(1, (g.nx, g.ny))
  phif = phi0

  q0 = @. sin(k*x)
  qf = q0

  phiforcing = @. 0.5*im*phi0*q0 # holds as long as phi=constant or phi=phi(x) with nu=0 and eta=0
  phiforcingh = fft(phiforcing)

  qforcing = @. kap*q0
  qforcingh = rfft(qforcing)

  function calcF!(Fc, Fr, t, s, v, p, g)
    Fc .= phiforcingh
    Fr .= qforcingh
    nothing
  end

  prob = NIWQG.Problem(nx=nx, Lx=Lx, dt=dt, kap=kap, nkap=0, nu=0.01, nnu=1, calcF=calcF!, stepper=stepper)
  NIWQG.set_q!(prob, q0)
  NIWQG.set_phi!(prob, phi0)

  stepforward!(prob, nsteps)
  NIWQG.updatevars!(prob)

  isapprox(prob.vars.q, qf, rtol=rtol_niwqg) && isapprox(prob.vars.phi, phif, rtol=rtol_niwqg)
end

"Test phi advection, refraction, and dispersion terms."
function test_niwqg_nonlinear2(; stepper="RK4", nk=2, nl=3, nx=32, Lx=2π, dt=0.01, nsteps=100, 
                               nu=0.01, eta=0.2, kap=0.02)
  k = 2π/Lx * nk
  l = 2π/Lx * nl
  g = TwoDGrid(nx, Lx)
  x, y = gridpoints(g)

  phi0 = @. exp(im*l*y)
  q0 = @. sin(k*x)
  V0 = @. -cos(k*x)/k

  phiforcing = @. 0.5*im*phi0*q0 + im*l*phi0*V0 + 0.5*im*eta*l^2*phi0 + nu*phi0
  phiforcingh = fft(phiforcing)

  qforcing = @. kap*q0
  qforcingh = rfft(qforcing)

  function calcF!(Fc, Fr, t, s, v, p, g)
    @. Fc = phiforcingh
    @. Fr = qforcingh
    nothing
  end

  prob = NIWQG.Problem(nx=nx, Lx=Lx, dt=dt, eta=eta, kap=kap, nkap=0, nu=nu, nnu=0, calcF=calcF!, stepper=stepper)
  NIWQG.set_q!(prob, q0)
  NIWQG.set_phi!(prob, phi0)

  stepforward!(prob, nsteps)
  NIWQG.updatevars!(prob)

  isapprox(prob.vars.q, q0, rtol=rtol_niwqg) && isapprox(prob.vars.phi, phi0, rtol=rtol_niwqg)
end

"Test wave PV."
function test_niwqg_wavepv(; stepper="RK4", nk=2, nx=32, Lx=2π, f=1, dt=0.1, nsteps=100, nu=0.01)
  k = 2π/Lx * nk
  g = TwoDGrid(nx, Lx)
  x, y = gridpoints(g)

  phi0 = @. cos(k*x)
  phiforcing = @. nu*phi0 + 0.125*im*k^2/f*(cos(k*x) + cos(3k*x))
  phiforcingh = fft(phiforcing)

  function calcF!(Fc, Fr, t, s, v, p, g)
    Fc .= phiforcingh
    nothing
  end

  prob = NIWQG.Problem(nx=nx, Lx=Lx, dt=dt, f=f, nu=nu, nnu=0, calcF=calcF!, stepper=stepper)
  NIWQG.set_phi!(prob, phi0)
  stepforward!(prob, nsteps)
  NIWQG.updatevars!(prob)

  isapprox(prob.vars.phi, phi0, rtol=rtol_niwqg)
end

"Test wave PV. Will fail if `calczetah!` is wrong."
function test_niwqg_calczetah(; nk=2, nl=2, nx=32, Lx=2π, f=0.2)
  k = 2π/Lx * nk
  l = 2π/Lx * nl

  prob = NIWQG.Problem(nx=nx, Lx=Lx, f=f)
  x, y = gridpoints(prob.grid)

  phi = @. cos(k*x) + im*cos(l*y)
  q = zeros(nx, nx)
  qw = @. -1/(2f) * (k^2*cos(2k*x) + l^2*cos(2l*y)) - k*l/f * sin(k*x) * sin(l*y)
  zeta1 = @. q - qw

  phih = fft(phi)
  qh = rfft(q)
  zetah1 = rfft(zeta1)

  zetah2 = @. 0*zetah1 # initialize
  NIWQG.calczetah!(zetah2, qh, phih, phi, prob.vars, prob.params, prob.grid)
  zeta2 = irfft(zetah2, nx)

  isapprox(zeta1, zeta2)
end

function test_niwqg_energetics(; nk=2, nl=3, nx=256, f=1, eta=0.4)
  prob = NIWQG.Problem(nx=nx, f=f, eta=eta)

  k = nk * 2π/prob.grid.Lx
  l = nl * 2π/prob.grid.Ly
  x, y = gridpoints(prob.grid)

  lam = sqrt(eta/f) # eta = f*lam^2

  phi0 = @. exp(im*k*x) 
  q0 = @. l*cos(l*y) # U = - q_y = -sin(l*y); ke = U^2/2 = 1/4

  action0 = 1/(2f) 
  ke0 = 1/4
  pe0 = lam^2/4 * k^2
  e0 = ke0 + pe0

  NIWQG.set_phi!(prob, phi0)
  NIWQG.set_q!(prob, q0)

  ( isapprox(NIWQG.waveaction(prob), action0)
    && isapprox(NIWQG.qgke(prob), ke0)
    && isapprox(NIWQG.wavepe(prob), pe0)
    && isapprox(NIWQG.coupledenergy(prob), e0)
  )
end

function test_niwqg_hyperdissipation_q(; kap=0.1, nkap=1, nk=2, nx=32, Lx=2π, nsteps=1, stepper="RK4")
  k = nk * 2π/Lx
  τ = 1/(kap*k^2nkap)
  dt = 0.001*τ
  tf = dt*nsteps

  prob = NIWQG.Problem(nx=nx, Lx=Lx, kap=kap, nkap=nkap, stepper=stepper, dt=dt)
  x, y = gridpoints(prob.grid)

  q0 = @. cos(k*x)
  qf = @. exp(-kap*k^2nkap*tf) * q0

  NIWQG.set_q!(prob, q0)
  stepforward!(prob, nsteps)
  NIWQG.updatevars!(prob)

  isapprox(qf, prob.vars.q, rtol=1e-15*nx^2*nsteps)
end


function test_niwqg_hyperdissipation_phi(; nu=0.1, nnu=1, nk=2, nx=64, Lx=2π, nsteps=1, stepper="RK4", eta=0.0)
  k = nk * 2π/Lx
  τ = 1/(nu*k^2nnu)
  dt = 0.01*τ
  tf = dt*nsteps

  prob = NIWQG.Problem(nx=nx, Lx=Lx, nu=nu, nnu=nnu, stepper=stepper, eta=eta, dt=dt)
  x, y = gridpoints(prob.grid)

  phi0 = @. exp(im*k*x)
  phif = @. exp(-im*0.5*eta*k^2*tf - nu*k^2nnu*tf) * phi0

  NIWQG.set_phi!(prob, phi0)
  stepforward!(prob, nsteps)
  NIWQG.updatevars!(prob)

  isapprox(phif, prob.vars.phi, rtol=1e-15*nx^2*nsteps)
end

function test_niwqg_exactnonlinearsolution(; U=1.1, a=0.9, nu=0.1, nk=2, nx=64, Lx=2π, nsteps=1, f=0.2, 
                                           stepper="ETDRK4") 
  k = nk * 2π/Lx
  τ = 1/(nu*k^2)
  dt = 0.01*τ
  tf = dt*nsteps
  
  prob = NIWQG.Problem(nx=nx, Lx=Lx, nu=nu, nnu=1, kap=nu, nkap=1, stepper=stepper, eta=0, f=f, dt=dt)
  x, y = gridpoints(prob.grid)

  phi0 = @. U*(1 + a*exp(im*k*x))
  q0 = @. -U^2*a*k^2/(2f) * cos(k*x)

  phif = @. U*(1 + a*exp(-nu*k^2*tf + im*k*x))
  qf = @. exp(-nu*k^2*tf)*q0

  NIWQG.set_phi!(prob, phi0)
  NIWQG.set_q!(prob, q0)

  stepforward!(prob, nsteps)
  NIWQG.updatevars!(prob)

  isapprox(phif, prob.vars.phi, rtol=1e-15*nx^2*nsteps) && isapprox(qf, prob.vars.q, rtol=1e-15*nx^2*nsteps)
end 

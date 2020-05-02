using PyPlot, FourierFlows, Printf

using Random: seed!

import GeophysicalFlows.TwoDNavierStokes
import GeophysicalFlows.TwoDNavierStokes: energy, enstrophy, dissipation, work, drag

dev = CPU()    # Device (CPU/GPU)
nothing # hide

 n, L  = 256, 2π             # grid resolution and domain length
 ν, nν = 1e-7, 2             # hyperviscosity coefficient and order
 μ, nμ = 1e-1, 0             # linear drag coefficient
dt, tf = 0.005, 0.2/μ        # timestep and final time
    nt = round(Int, tf/dt)   # total timesteps
    ns = 4                   # how many intermediate times we want to plot
nothing # hide

kf, dkf = 12.0, 2.0     # forcing central wavenumber, wavenumber width
ε = 0.1                 # energy injection rate

gr   = TwoDGrid(dev, n, L)
x, y = gridpoints(gr)

Kr = ArrayType(dev)([ gr.kr[i] for i=1:gr.nkr, j=1:gr.nl])

forcingcovariancespectrum = @. exp(-(sqrt(gr.Krsq)-kf)^2/(2*dkf^2))
forcingcovariancespectrum[gr.Krsq .< 2.0^2 ] .= 0  # making sure that focing has no power for low wavenumbers
forcingcovariancespectrum[gr.Krsq .> 20.0^2 ] .= 0 # making sure that focing has no power for high wavenumbers
forcingcovariancespectrum[Kr .< 2π/L] .= 0
ε0 = FourierFlows.parsevalsum(forcingcovariancespectrum.*gr.invKrsq/2.0, gr)/(gr.Lx*gr.Ly)
forcingcovariancespectrum .= ε/ε0 * forcingcovariancespectrum # normalize forcing to inject energy ε

seed!(1234)
nothing # hide

function calcF!(Fh, sol, t, cl, v, p, g)
  eta = ArrayType(dev)(exp.(2π*im*rand(typeof(gr.Lx), size(sol)))/sqrt(cl.dt))
  eta[1, 1] = 0
  @. Fh = eta*sqrt(forcingcovariancespectrum)
  nothing
end
nothing # hide

prob = TwoDNavierStokes.Problem(nx=n, Lx=L, ν=ν, nν=nν, μ=μ, nμ=nμ, dt=dt, stepper="RK4",
                        calcF=calcF!, stochastic=true, dev=dev)
nothing # hide

sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
nothing # hide

TwoDNavierStokes.set_zeta!(prob, 0*x)

E = Diagnostic(energy,      prob, nsteps=nt) # energy
R = Diagnostic(drag,        prob, nsteps=nt) # dissipation by drag
D = Diagnostic(dissipation, prob, nsteps=nt) # dissipation by hyperviscosity
W = Diagnostic(work,        prob, nsteps=nt) # work input by forcing
diags = [E, D, W, R] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
nothing # hide

function makeplot(prob, diags)
  TwoDNavierStokes.updatevars!(prob)
  E, D, W, R = diags

  t = round(μ*cl.t, digits=2)
  sca(axs[1]); cla()
  pcolormesh(x, y, v.zeta)
  xlabel(L"$x$")
  ylabel(L"$y$")
  title("\$\\nabla^2\\psi(x,y,\\mu t= $t )\$")
  axis("square")

  sca(axs[3]); cla()

  i₀ = 1
  dEdt = (E[(i₀+1):E.i] - E[i₀:E.i-1])/cl.dt #numerical first-order approximation of energy tendency
  ii = (i₀):E.i-1
  ii2 = (i₀+1):E.i

  total = W[ii2] - D[ii] - R[ii]        # Stratonovich interpretation
  residual = dEdt - total

  plot(μ*E.t[ii], W[ii2], label=L"work ($W$)")
  plot(μ*E.t[ii], ε .+ 0*E.t[ii], "--", label=L"ensemble mean  work ($\langle W\rangle $)")
  plot(μ*E.t[ii], -D[ii], label="dissipation (\$D\$)")
  plot(μ*E.t[ii], -R[ii], label=L"drag ($D=2\mu E$)")
  plot(μ*E.t[ii], 0*E.t[ii], "k:", linewidth=0.5)
  ylabel("Energy sources and sinks")
  xlabel(L"$\mu t$")
  legend(fontsize=10)

  sca(axs[2]); cla()
  plot(μ*E.t[ii], total[ii], label=L"computed $W-D$")
  plot(μ*E.t[ii], dEdt, "--k", label=L"numerical $dE/dt$")
  ylabel(L"$dE/dt$")
  xlabel(L"$\mu t$")
  legend(fontsize=10)

  sca(axs[4]); cla()
  plot(μ*E.t[ii], residual, "c-", label=L"residual $dE/dt$ = computed $-$ numerical")
  xlabel(L"$\mu t$")
  legend(fontsize=10)
end
nothing # hide

startwalltime = time()
for i = 1:ns
  stepforward!(prob, diags, round(Int, nt/ns))
  TwoDNavierStokes.updatevars!(prob)
  cfl = cl.dt*maximum([maximum(v.u)/g.dx, maximum(v.v)/g.dy])

  log = @sprintf("step: %04d, t: %.1f, cfl: %.3f, walltime: %.2f min", cl.step, cl.t,
        cfl, (time()-startwalltime)/60)

  println(log)
end

fig, axs = subplots(ncols=2, nrows=2, figsize=(12, 8), dpi=200)
makeplot(prob, diags)
gcf() # hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl


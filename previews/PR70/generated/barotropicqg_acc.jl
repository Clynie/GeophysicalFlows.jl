using FourierFlows, PyPlot, Printf

using FFTW: irfft

import GeophysicalFlows.BarotropicQG
import GeophysicalFlows.BarotropicQG: energy, meanenergy, enstrophy, meanenstrophy

dev = CPU()     # Device (CPU/GPU)
nothing # hide

nx  = 128      # 2D resolution = nx^2
stepper = "ETDRK4"   # timestepper
dt  = 1e-1     # timestep
nsteps = 10000 # total number of time-steps
nsubs  = 2500  # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)
nothing # hide

Lx = 2π        # domain size
 ν = 4e-15     # viscosity
nν = 4         # viscosity order
f0 = -1.0      # Coriolis parameter
 β = 1.4015    # the y-gradient of planetary PV
 μ = 1e-2      # linear drag
 F = 0.0012    # normalized wind stress forcing on domain-averaged zonal flow U(t) flow
nothing # hide

topoPV(x, y) = @. 2*cos(4x)*cos(4y)
nothing # hide

calcFU(t) = F
nothing # hide

prob = BarotropicQG.Problem(nx=nx, Lx=Lx, f0=f0, β=β, eta=topoPV,
                  calcFU=calcFU, ν=ν, nν=nν, μ=μ, dt=dt, stepper=stepper, dev=dev)
nothing # hide

sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x, y = gridpoints(g)
nothing # hide

BarotropicQG.set_zeta!(prob, 0*x)

E = Diagnostic(energy, prob; nsteps=nsteps)
Q = Diagnostic(enstrophy, prob; nsteps=nsteps)
Emean = Diagnostic(meanenergy, prob; nsteps=nsteps)
Qmean = Diagnostic(meanenergy, prob; nsteps=nsteps)
diags = [E, Emean, Q, Qmean]
nothing # hide

filepath = "."
plotpath = "./plots_barotropicqgtopography"
plotname = "snapshots"
filename = joinpath(filepath, "barotropicqgtopography.jld2")
nothing # hide

if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end
nothing # hide

get_sol(prob) = sol # extracts the Fourier-transformed solution
get_u(prob) = irfft(im*g.lr.*g.invKrsq.*sol, g.nx)
out = Output(prob, filename, (:sol, get_sol), (:u, get_u))
nothing # hide

function plot_output(prob, fig, axs; drawcolorbar=false)

  sol, v, p, g = prob.sol, prob.vars, prob.params, prob.grid
  BarotropicQG.updatevars!(prob)

  sca(axs[1])
  pcolormesh(x, y, v.q)
  axis("square")
  xticks(-2:2)
  yticks(-2:2)
  title(L"$\nabla^2\psi + \eta$")
  if drawcolorbar==true
    colorbar()
  end

  sca(axs[2])
  cla()
  plot(μ*E.t[1:E.i], E.data[1:E.i], label=L"$E_{\psi}$")
  plot(μ*E.t[1:Emean.i], Emean.data[1:Emean.i], label=L"$E_U$")

  xlabel(L"\mu t")
  ylabel("energy")
  legend()

  sca(axs[3])
  cla()
  plot(μ*Q.t[1:Q.i], Q.data[1:Q.i], label=L"$Q_{\psi}$")
  plot(μ*Qmean.t[1:Qmean.i], Qmean.data[1:Qmean.i], label=L"$Q_U$")
  xlabel(L"\mu t")
  ylabel("potential enstrophy")
  legend()
  tight_layout(w_pad=0.1)
end
nothing # hide

startwalltime = time()

while cl.step < nsteps
  stepforward!(prob, diags, nsubs)

  cfl = cl.dt*maximum([maximum(v.U.+v.u)/g.dx, maximum(v.v)/g.dy])

  log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Q: %.4f, walltime: %.2f min",
    cl.step, cl.t, cfl, E.data[E.i], Q.data[Q.i], (time()-startwalltime)/60)

  println(log)
end
println("finished")

fig, axs = subplots(ncols=3, nrows=1, figsize=(15, 4), dpi=200)
plot_output(prob, fig, axs; drawcolorbar=true)
gcf() # hide

savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), cl.step)
savefig(savename)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl


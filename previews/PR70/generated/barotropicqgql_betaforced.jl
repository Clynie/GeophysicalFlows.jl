using FourierFlows, PyPlot, Statistics, Printf, Random

import GeophysicalFlows.BarotropicQGQL
import GeophysicalFlows.BarotropicQGQL: energy, enstrophy

import FFTW: irfft
import Random: seed!
import Statistics: mean

nx = 128       # 2D resolution = nx^2
stepper = "FilteredRK4"   # timestepper
dt  = 0.05     # timestep
nsteps = 8000  # total number of time-steps
nsubs  = 2000  # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)
nothing # hide

Lx = 2π        # domain size
 ν = 0.0       # viscosity
nν = 1         # viscosity order
 β = 10.0      # planetary PV gradient
 μ = 0.01      # bottom drag
nothing # hide

kf, dkf = 14.0, 1.5     # forcing wavenumber and width of forcing ring in wavenumber space
ε = 0.001               # energy input rate by the forcing

gr  = TwoDGrid(nx, Lx)

x, y = gridpoints(gr)
Kr = [ gr.kr[i] for i=1:gr.nkr, j=1:gr.nl]

forcingcovariancespectrum = @. exp(-(sqrt(gr.Krsq)-kf)^2/(2*dkf^2))
@. forcingcovariancespectrum[gr.Krsq < 2.0^2 ] .= 0
@. forcingcovariancespectrum[gr.Krsq > 20.0^2 ] .= 0
forcingcovariancespectrum[Kr .< 2π/Lx] .= 0
ε0 = FourierFlows.parsevalsum(forcingcovariancespectrum.*gr.invKrsq/2.0, gr)/(gr.Lx*gr.Ly)
forcingcovariancespectrum .= ε/ε0 * forcingcovariancespectrum  # normalization so that forcing injects energy ε per domain area per unit time

seed!(1234) # reset of the random number generator for reproducibility
nothing # hide

function calcF!(Fh, sol, t, cl, v, p, g)
  ξ = exp.(2π*im*rand(Float64, size(sol)))/sqrt(cl.dt)
  ξ[1, 1] = 0
  @. Fh = ξ*sqrt(forcingcovariancespectrum)
  Fh[abs.(Kr) .== 0] .= 0
  nothing
end
nothing # hide

prob = BarotropicQGQL.Problem(nx=nx, Lx=Lx, beta=β, nu=ν, nnu=nν, mu=μ, dt=dt,
                              stepper=stepper, calcF=calcF!, stochastic=true)
nothing # hide

sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
nothing # hide

BarotropicQGQL.set_zeta!(prob, 0*x)
nothing # hide

E = Diagnostic(energy, prob; nsteps=nsteps)
Z = Diagnostic(enstrophy, prob; nsteps=nsteps)
nothing # hide

function zetaMean(prob)
  sol = prob.sol
  sol[1, :]
end

zMean = Diagnostic(zetaMean, prob; nsteps=nsteps, freq=10)  # the zonal-mean vorticity
nothing # hide

diags = [E, Z, zMean] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
nothing # hide

filepath = "."
plotpath = "./plots_forcedbetaQLturb"
plotname = "snapshots"
filename = joinpath(filepath, "forcedbetaQLturb.jld2")
nothing # hide

if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end
nothing # hide

get_sol(prob) = sol # extracts the Fourier-transformed solution
get_u(prob) = irfft(im*g.l.*g.invKrsq.*sol, g.nx)
out = Output(prob, filename, (:sol, get_sol), (:u, get_u))
nothing # hide

function plot_output(prob, fig, axs; drawcolorbar=false)
  sol, v, p, g = prob.sol, prob.vars, prob.params, prob.grid
  BarotropicQGQL.updatevars!(prob)

  sca(axs[1])
  cla()
  pcolormesh(x, y, v.zeta .+ v.Zeta)
  axis("square")
  xticks(-3:1:3)
  yticks(-3:1:3)
  title(L"vorticity $\zeta = \partial_x v - \partial_y u$")
  if drawcolorbar==true
    colorbar()
  end

  sca(axs[2])
  cla()
  contourf(x, y, v.psi)
  if maximum(abs.(v.psi))>0
    contour(x, y, v.psi, colors="k")
  end
  axis("square")
  xticks(-3:1:3)
  yticks(-3:1:3)
  title(L"streamfunction $\psi$")
  if drawcolorbar==true
    colorbar()
  end

  sca(axs[3])
  cla()
  plot(Array(transpose(mean(v.Zeta, dims=1))), y[1, :])
  plot(0*y[1, :], y[1, :], "k--")
  yticks(-3:1:3)
  ylim(-Lx/2, Lx/2)
  xlim(-3, 3)
  title(L"zonal mean $\zeta$")

  sca(axs[4])
  cla()
  plot(Array(mean(transpose(v.U), dims=2)), y[1, :])
  plot(0*y[1, :], y[1, :], "k--")
  yticks(-3:1:3)
  ylim(-Lx/2, Lx/2)
  xlim(-0.5, 0.5)
  title(L"zonal mean $u$")

  sca(axs[5])
  cla()
  plot(μ*E.t[1:E.i], E.data[1:E.i], label="energy")
  xlabel(L"\mu t")
  legend()

  sca(axs[6])
  cla()
  plot(μ*Z.t[1:Z.i], Z.data[1:E.i], label="enstrophy")
  xlabel(L"\mu t")
  legend()
end
nothing # hide

startwalltime = time()

while cl.step < nsteps
  stepforward!(prob, diags, nsubs)

  BarotropicQGQL.updatevars!(prob)

  cfl = cl.dt*maximum([maximum(v.v)/g.dy, maximum(v.u+v.U)/g.dx])
  log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Q: %.4f, walltime: %.2f min",
    cl.step, cl.t, cfl, E.data[E.i], Z.data[Z.i],
    (time()-startwalltime)/60)

  println(log)
end
println("finished")

fig, axs = subplots(ncols=3, nrows=2, figsize=(14, 8), dpi=200)
plot_output(prob, fig, axs; drawcolorbar=false)
gcf() #hide

savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), cl.step)
savefig(savename)
nothing #hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl


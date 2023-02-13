# Produce the figure used in README.md

using AutoregressiveModels, CSV, CairoMakie, ConfidenceBands
using LocalProjections: datafile

set_theme!()
update_theme!(Axis=(rightspinevisible=false, topspinevisible=false, spinewidth=0.7,
    xgridvisible=false, titlefont="Helvetica", titlegap=5, xtickwidth=0.7, ytickwidth=0.7,
    xticksize=5, yticksize=5), font="Helvetica", fontsize=12, figure_padding=5)

data = CSV.File(datafile(:gk))
names = (:logcpi, :logip, :gs1, :ebp)
r = fit(VARProcess, data, names, 12, choleskyresid=true, adjust_dofr=false)
irf = impulse(r, 3, 37, choleskyshock=true)

fillirf!(x) = impulse!(x.out, x.r, 3, choleskyshock=true)
ndraw = 10000
bootirfs = Array{Float64, 3}(undef, 4, 37, ndraw)

bootstrap!(bootirfs=>fillirf!, r, initialindex=1, drawresid=iidresiddraw!)
boot2 = view(bootirfs, 2, :, :)
lb, ub, pwlevel = confint(SuptQuantileBootBand(), boot2, level=0.68)

res = 72 .* (6.5, 3.5)
fig = Figure(; resolution=res, backgroundcolor=:transparent)
ax = Axis(fig[1, 1], xlabel="Month", ylabel="Impulse Response", backgroundcolor=:transparent)
band!(ax, 0:36, lb, ub, color=(:red, 0.1))
lines!(ax, 0:36, view(irf, 2, :), color=:red)
lines!(ax, 0:36, lb, color=:red, linewidth=0.5)
lines!(ax, 0:36, ub, color=:red, linewidth=0.5)
ax.xticks = 0:12:36
save("docs/src/assets/readmeexample.svg", fig, pt_per_unit=1)
fig

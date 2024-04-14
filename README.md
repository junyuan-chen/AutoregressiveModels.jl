# AutoregressiveModels.jl

*Essential toolkits for working with autoregressive models*

[![CI-stable][CI-stable-img]][CI-stable-url]
[![codecov][codecov-img]][codecov-url]
[![PkgEval][pkgeval-img]][pkgeval-url]

[CI-stable-img]: https://github.com/junyuan-chen/AutoregressiveModels.jl/actions/workflows/CI-stable.yml/badge.svg?branch=main
[CI-stable-url]: https://github.com/junyuan-chen/AutoregressiveModels.jl/actions/workflows/CI-stable.yml

[codecov-img]: https://codecov.io/gh/junyuan-chen/AutoregressiveModels.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/junyuan-chen/AutoregressiveModels.jl

[pkgeval-img]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/A/AutoregressiveModels.svg
[pkgeval-url]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/A/AutoregressiveModels.html

[AutoregressiveModels.jl](https://github.com/junyuan-chen/AutoregressiveModels.jl)
is a Julia package that provides essential toolkits for working with autoregressive models.
Performance and reusability is prioritized over comprehensive coverage of functionalities,
as a main goal of the package is to provide support
for other packages with more specialized purposes.
At this moment, the main focus is on vector autoregressions (VAR).
Estimation of factor models is implemented for balanced panel data
following Stock and Watson (2016).
Some basic support for the autoregressive-moving-average (ARMA) models is also included.

## Example Usage

To illustrate what the package offers,
here is an example of estimating the impulse responses
based on structural vector autoregressions (SVAR)
and producing a simultaneous confidence band with bootstrap.
Details for individual functions may be found
from docstrings in the help mode of Julia REPL.

### Impulse Responses from Structural VAR

The example below reproduces one application from Montiel Olea and Plagborg-Møller (2019).
The data used are from Gertler and Karadi (2015).

#### Step 1: Model Specification and Point Estimates

```julia
using AutoregressiveModels, CSV, ConfidenceBands
using LocalProjections: datafile # Only needed for the data file

# Load a prepared data file from Gertler and Karadi (2015)
data = CSV.File(datafile(:gk))
# Specify the variables for VAR (the order matters)
names = (:logcpi, :logip, :gs1, :ebp)
# Estimate VAR(12) with OLS and conduct Cholesky factorization for identification
r = fit(VARProcess, data, names, 12, choleskyresid=true, adjust_dofr=false)
# Compute point estimates of impulse responses (37 horizons) to the structural shock (3)
irf = impulse(r, 3, 37, choleskyshock=true)
```

#### Step 2: Bootstrap Confidence Band

A flexible autoregressive bootstrap framework is defined via `bootstrap!`
and can be used to produce the draws of estimates for
`SuptQuantileBootBand()` implemented in
[ConfidenceBands.jl](https://github.com/junyuan-chen/ConfidenceBands.jl):

```julia
# Define how the bootstrap statistics are computed
# See the docstring of bootstrap! for explanations
fillirf!(x) = impulse!(x.out, x.r, 3, choleskyshock=true)
ndraw = 10000
# Preallocate an output array for statistics computed over the bootstrap iterations
bootirfs = Array{Float64, 3}(undef, 4, 37, ndraw)

# Specify the bootstrap procedure
bootstrap!(bootirfs=>fillirf!, r, initialindex=1, drawresid=iidresiddraw!)
# Produce a confidence band from the result
boot2 = view(bootirfs, 2, :, :)
lb, ub, pwlevel = confint(SuptQuantileBootBand(), boot2, level=0.68)
```

#### Step 3: Visualization

Here is a plot for the results with the complete script located
[here](https://raw.githubusercontent.com/junyuan-chen/AutoregressiveModels.jl/main/docs/src/plots/readmeexample.jl):

<p align="center">
  <img src="https://raw.githubusercontent.com/junyuan-chen/AutoregressiveModels.jl/main/docs/src/assets/readmeexample.svg" height="252"><br>
</p>

## References

**Gertler, Mark, and Peter Karadi.** 2015.
"Replication Data for: Monetary Policy Surprises, Credit Costs, and Economic Activity."
*American Economic Association* [publisher], Inter-university Consortium for Political and Social Research [distributor]. https://doi.org/10.3886/E114082V1.

**Montiel Olea, José Luis and Mikkel Plagborg-Møller.** 2019.
"Simultaneous Confidence Bands: Theory, Implementation, and an Application to SVARs."
*Journal of Applied Econometrics* 34 (1): 1-17.

**Stock, James H. and Mark W. Watson.** 2016.
"Chapter 8---Dynamic Factor Models, Factor-Augmented Vector Autoregressions, and Structural Vector Autoregressions in Macroeconomics."
In *Handbook of Macroeconomics*, Vol. 2A,
edited by John B. Taylor and Harald Uhlig, 415-525. Amsterdam: Elsevier.

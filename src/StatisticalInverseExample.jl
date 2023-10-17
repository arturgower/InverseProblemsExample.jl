# Importing needed libraries
using MultipleScattering
using LinearAlgebra
using Plots
using NumericalIntegration
using Random
using NumericalIntegration
using LaTeXStrings

# Setting font sizes and seed
scalefontsizes(1.6)
Random.seed!(73)

# Prior choice: uniform (0) or log-normal (1)
log_normal = 1

# Well-posed
ω = 1.0

# Ill-posed (uncomment line below)
ω = 4.7

# Radius space discretisation
Na = 400
amax = 10.0
amin = amax/Na
a = LinRange(amin, amax, Na)

# Picking random radius
a_star = 1.9
a_star = 4.9

# Parameters for the likelihoood
σf = 0.1
order = 5

# Building prior
if log_normal == 1
    μ1 = 0.6
    σ1 = 0.06
    μ2 = 1.6
    σ2 = 0.03
    ln = exp.(-(log.(a) .- μ1).^2 ./ σ1^2) + exp.(-(log.(a) .- μ2).^2 ./ σ2^2)
    prior = ln ./ integrate(a, ln)
    plot(a, prior, linewidth = 2, label = "")
else
    prior = zeros(Na)
    for i in 1:length(a)
        r = a[i]
        if r >= 0.5 && r < 3.5
            prior[i] = 1/3
        end
    end
    plot(ω .* a, prior, linewidth = 2, label = "")
end
plot!(xlabel = L"a [\mu m]")
plot!(ylabel = L"p(a)")
plot!(xlims = [1.0,6.0])
vline!([1.9], linewidth = 2, label = L"a_1")
vline!([4.9], linewidth = 2, label = L"a_2")
savefig("Prior_with_radii.png")

# Defining properties of the cylinder and host medium
host_medium = Acoustic(1.0, 1.0, 2)
cylinder_medium = Acoustic(Inf, Inf + 0.0im, 2)
cylinder_shape = Circle(a_star)
cylinder = Particle(cylinder_medium, cylinder_shape)

# Making measurement of far field pattern with 1% random error
T = diag(t_matrix(cylinder, host_medium, ω, order))
f = abs(sum(T))
f = log(f + (rand() - 0.5) * (f / 50))

# Building likelihood
g = zeros(length(a))
for i in 1:length(a)
    cyl = Particle(cylinder_medium, Circle(a[i]))
    T = diag(t_matrix(cyl, host_medium, ω, order))
    g[i] = log(abs(sum(T)))
end
gauss = exp.(-(g .- f).^2 ./ σf^2)
likelihood = gauss ./ integrate(a, gauss)
plot(a, likelihood)
#savefig("Likelihood")

# Computing posterior
mult = prior .* likelihood
evidence = integrate(a, mult)
posterior = mult ./ evidence

# Computing prior mean
prior_mean = integrate(a, prior .* a)

# Computing likelihood mean and maximum
mean_likelihood = integrate(a, likelihood .* a)
max_likelihood = a[findmax(likelihood)[2]]

# Computing posterior mean, maximum and variance
mean = integrate(a, posterior .* a)
square_mean = integrate(a, posterior .* (a.^2))
max_posterior = a[findmax(posterior)[2]]
var = square_mean - mean^2

# Plotting likelihood
plot(ω .* a, likelihood, label = "", linewidth = 2)
vline!([ω * a_star], label = "Real value", linewidth = 2)
vline!([ω * max_likelihood], label = "Maximum of likelihood", linewidth = 2)
vline!([ω * mean_likelihood], label = "Mean of likelihood", linewidth = 2)
plot!(xlabel = L"k a")
plot!(ylabel = L"p(y | a)")
#savefig("Likelihood.png")

# Plotting posterior
plot(ω .* a, posterior, label = "Posterior", linewidth = 2, legend = :outertopright)
vline!([ω * a_star], label = "Real value", linewidth = 2)
vline!([ω * max_posterior], label = "Maximum of posterior", linewidth = 2)
vline!([ω * mean], label = "Mean of posterior", linewidth = 2)
plot!(xlabel = L"k a")
plot!(ylabel = L"p(a | y)")
plot!(legendfont=font(12))
plot!(xlims = [4.0,28.0])
#savefig("Posterior_well_posed")
#savefig("Posterior_22_vs_78.png")

#plot!(xlims = [ω * 0.1,ω * 1.8])

# Full likelihood plot
gs = zeros(Na)
for i in 1:length(a)
    cylinder_shape = Circle(a[i])
    cylinder = Particle(cylinder_medium, cylinder_shape)
    T = diag(t_matrix(cylinder, host_medium, ω, order))
    gs[i] = abs(sum(T))
end
error = [5*0.1 for i in 1:Na]
plot(ω .* a, gs, linewidth = 2, label = L"f(a)")
#plot(ω .* a, gs, linewidth = 2, ribbon = error, label = L"f(a) \pm 5 \times \sigma_y")
plot!(xlabel = L"k a")
#plot!(ylabel = L"g(a)")
plot!(legend = :bottomright)
hline!([exp(f)], label = "Measure y", linewidth = 2)
plot!(ω .* a, 4 .* prior, linewidth = 2, label = "Size dist.")
#savefig("f(a)_and_measure.png")
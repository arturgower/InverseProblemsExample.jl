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
ω = 5.0

# Radius space discretisation
Na = 400
amax = 10.0
amin = amax/Na
a = LinRange(amin, amax, Na)

# Picking random radius
a_star = (3 * rand()) + 0.5
#a_star = 0.7

# Parameters for the likelihoood
σf = 0.1
order = 5

# Building prior
if log_normal == 1
    μ = 0.5
    σ = 0.5
    ln = exp.(-(log.(a) .- μ).^2 ./ σ^2)
    prior = ln ./ integrate(a, ln)
    plot(ω .* a, prior, linewidth = 2, label = "Prior")
else
    prior = zeros(Na)
    for i in 1:length(a)
        r = a[i]
        if r >= 0.5 && r < 3.5
            prior[i] = 1/3
        end
    end
    plot(ω .* a, prior, linewidth = 2, label = "Prior")
end
plot!(xlabel = L"k a")
plot!(ylabel = L"p(a)")
plot!(legendfont = font(10))
#savefig("Prior")

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
plot(ω .* a, likelihood, label = "Likelihood", linewidth = 2)
vline!([ω * a_star], label = "Real value", linewidth = 2)
vline!([ω * max_likelihood], label = "Maximum of likelihood", linewidth = 2)
vline!([ω * mean_likelihood], label = "Mean of likelihood", linewidth = 2)
plot!(xlabel = L"k a")
plot!(ylabel = L"p(f | a)")
plot!(legendfont = font(10))

# Plotting posterior
plot(ω .* a, posterior, label = "Posterior", linewidth = 2)
vline!([ω * a_star], label = "Real value", linewidth = 2)
vline!([ω * max_posterior], label = "Maximum of posterior", linewidth = 2)
vline!([ω * mean], label = "Mean of posterior", linewidth = 2)
plot!(xlabel = L"k a")
plot!(ylabel = L"p(a | f)")
plot!(legendfont=font(12))
#savefig("Posterior_well_posed")
#savefig("Posterior_ill_posed")

#plot!(xlims = [ω * 0.1,ω * 1.8])

# Full likelihood plot
ω = 5.0
gs = zeros(Na)
for i in 1:length(a)
    cylinder_shape = Circle(a[i])
    cylinder = Particle(cylinder_medium, cylinder_shape)
    T = diag(t_matrix(cylinder, host_medium, ω, order))
    gs[i] = abs(sum(T))
end
error = [5*0.1 for i in 1:Na]
plot(ω .* a, gs, linewidth = 2, ribbon = error, label = L"g(a) \pm 5 \times \sigma_f")
plot!(xlabel = L"k a")
plot!(ylabel = L"g(a)")
plot!(legend = :bottomright)
hline!([exp(f)], label = "Measurement", linewidth = 2)
#savefig("g(a)_and_error.png")
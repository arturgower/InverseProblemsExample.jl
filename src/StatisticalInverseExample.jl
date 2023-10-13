using MultipleScattering
using LinearAlgebra
using Plots
using NumericalIntegration
using Random
using NumericalIntegration

#Prior choice
log_normal = 0

a_star = (3 * rand()) + 0.5

#omega = 1.0

#a_star = 1.75 
#a_star = 7.0
#a_star = 8.0
#a_star = 0.1
#a_star = 0.57

#ω = 0.01

#a_star = 1.85

#ω = 50.0

Na = 400
amax = 10.0
amin = amax/Na
μ = 0.5
σ = 0.5
σf = 0.01
order = 5
a = LinRange(amin, amax, Na)

if log_normal == 1
    ln = exp.(-(log.(a) .- μ).^2 ./ σ^2)
    prior = ln ./ integrate(a, ln)
    plot(a, prior)
else
    prior = zeros(Na)
    for i in 1:length(a)
        r = a[i]
        if r >= 0.5 && r < 3.5
            prior[i] = 1/3
        end
    end
    plot(a, prior)
end

ω = 1.0
host_medium = Acoustic(1.0, 1.0, 2)
cylinder_medium = Acoustic(Inf, Inf + 0.0im, 2)
cylinder_shape = Circle(a_star)
cylinder = Particle(cylinder_medium, cylinder_shape)

T = diag(t_matrix(cylinder, host_medium, ω, order))
f = abs(sum(T))
f = log(f + (rand() - 0.5) * (f / 50))

g = zeros(length(a))
for i in 1:length(a)
    cyl = Particle(cylinder_medium, Circle(a[i]))
    T = diag(t_matrix(cyl, host_medium, ω, order))
    g[i] = log(abs(sum(T)))
end

gauss = exp.(-(g .- f).^2 ./ σf^2)
likelihood = gauss ./ integrate(a, gauss)
plot(a, likelihood)

mult = prior .* likelihood
evidence = integrate(a, mult)
posterior = mult ./ evidence

prior_mean = integrate(a, prior .* a)
mean_likelihood = integrate(a, likelihood .* a)
max_likelihood = a[findmax(likelihood)[2]]

mean = integrate(a, posterior .* a)
square_mean = integrate(a, posterior .* (a.^2))
max_posterior = a[findmax(posterior)[2]]
var = square_mean - mean^2

plot(a, likelihood, label = "Likelihood")
vline!([a_star], label = "Real value")
vline!([max_likelihood], label = "Maximum of likelihood")
vline!([mean_likelihood], label = "Mean of likelihood")

plot(a, posterior, label = "Posterior")
vline!([a_star], label = "Real value")
vline!([max_posterior], label = "Maximum of posterior")
vline!([mean], label = "Mean of posterior")
if log_normal == 0
    plot!(xlims = [0.0,4.0])
end

#plot!(xlims = [6.0,8.0])
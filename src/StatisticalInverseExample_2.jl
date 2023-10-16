using MultipleScattering
using LinearAlgebra
using Plots
using NumericalIntegration
using Random
using SpecialFunctions
using NumericalIntegration

#Prior choice
log_normal = 0

a_star = (3 * rand()) + 0.5

Na = 400
amax = 10.0
amin = amax/Na
a = LinRange(amin, amax, Na)

Ny = 200
ymax = 20.0
y = LinRange(-ymax, ymax, Ny)
y_star = 2*ymax*(rand() - 0.5)

σf = 0.01
order = 5

if log_normal == 1
    μ = 0.5
    σ = 0.5
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
matrix = zeros(Complex, 2*order+1, 2*order+1)
for n in -order:order
    for m in -order:order
        matrix[n + order+1, m + order+1] = besselj(n-m, ω*y_star)*complex(0.0, 1.0)^(n-m)
    end
end
f = abs(sum(matrix * T))
f = log(f + (rand() - 0.5) * (f / 50))

h = zeros(length(a), length(y))
for i in 1:length(a)
    for j in 1:length(y)
        cyl = Particle(cylinder_medium, Circle(a[i]))
        T = diag(t_matrix(cyl, host_medium, ω, order))
        matrix = zeros(Complex, 2*order+1, 2*order+1)
        for n in -order:order
            for m in -order:order
                matrix[n + order+1, m + order+1] = besselj(n-m, ω*y[j])*complex(0.0, 1.0)^(n-m)
            end
        end
        h[i, j] = log(abs(sum(matrix * T)))
    end
end

gauss_1 = exp.(-(h .- f).^2 ./ σf^2)
gauss_2 = zeros(Na)
for i in 1:length(a)
    gauss_2[i] = integrate(y, gauss_1[i,:])
end
likelihood = gauss_2 ./ integrate(a, gauss_2)
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

using Images
using FileIO

using Plots
using Statistics
using LinearAlgebra


# using ImageCore, TestImages
# using Noise
# using ImageFiltering
# using FFTW
# using DSP


img = FileIO.load("images/flag.png")
gray_img=Gray.(img)
M = Float64.(gray_img)

i1, i2 = size(M)

sample_rate = 9;
M = M[1:sample_rate:i1,1:sample_rate:i2]

Gray.(M)

# function gauss_kernel(σ)
#        ker(x,y) = (1/(2π * σ)) * exp(- x^2 / σ^2 - y^2 / σ^2)
#     return ker
   
# end

const σ = 0.0

kernel(x,y) = (1/(2π * σ)) * exp(- x^2 / σ^2 - y^2 / σ^2)


kernel(x,y) = exp(- x^2 / (2.0)^2 - y^2 / (2.0)^2)

img_size = size(M)

A = zeros(img_size[1],img_size[2],img_size[1],img_size[2])

for i in CartesianIndices(A)
    A[i] = kernel(i[1]-i[3],i[2]-i[4])
end

l1 =  Int(round(img_size[1]/2))
l2 =  Int(round(img_size[2]/2))

sumkernal = sum(kernel(x,y) for x in -l1:l1, y in -l2:l2)

A = reshape(A,length(M),length(M)) ./ sumkernal

ϵ = 0.01
v = A * M[:] + ϵ .* randn(length(M))
# v = A * M[:] 

Mtrans = reshape(v,img_size...)


plot(Gray.(Mtrans))
savefig("images/blur-flag.png")


# invA = inv(A)

# the obvious solution is
vsol = A \ v
norm(A * vsol - v) / norm(v)

Msol = reshape(vsol, img_size...);

Gray.(Msol)


svdA = svd(A)
plot(svdA.S)

inds = findall(svdA.S .> 5ϵ)

Astable = svdA.U[:,inds] * Diagonal(svdA.S[inds]) * svdA.Vt[inds,:]
norm(Astable - A) / norm(A)

Ainv_stable = transpose(svdA.Vt[inds,:]) * Diagonal(1 ./ svdA.S[inds]) * transpose(svdA.U[:,inds])

norm(Ainv_stable * A - I) / length(A)
norm(A * Ainv_stable - I) / length(A)


vsol = Ainv_stable * v
norm(A * vsol - v) / norm(v)

Msol = reshape(vsol, img_size...);

Gray.(Msol)

plot(Gray.(Msol))
# savefig("images/recover-svd-flag.png")

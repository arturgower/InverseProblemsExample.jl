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


## Flag example
    img = FileIO.load("images/flag.png")
    gray_img = Gray.(img)
    M = Float64.(gray_img)

    i1, i2 = size(M)
    sample_rate = 12;
    M = M[1:sample_rate:i1,1:sample_rate:i2]

    plot(Gray.(M), axis = false, 
        xlab = "", ylab = "", 
        frame = :none
    )    

    # gr(size = (size(M)[2],size(M)[1]) .* 6)
    # plot(Gray.(M), xlims = (2,67), ylims = (2,44))
    # plot!(axis = false, 
    #     xlab = "", ylab = "", 
    #     frame = :none,
    #     title="", xguide ="",yguide ="")

    # savefig("images/small-flag.png", size = (53,80))

    # const σ = 0.0

    # kernel(x,y) = (1/(2π * σ)) * exp(- x^2 / σ^2 - y^2 / σ^2)

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

    plot(Gray.(Mtrans), axis = false, 
        xlab = "", ylab = "", 
        frame = :none
    )    
    savefig("images/blur-flag.png")

    # invA = inv(A)

    # the obvious solution is
    vsol = A \ v
    norm(A * vsol - v) / norm(v)

    Msol = reshape(vsol, img_size...);

    plot(Gray.(Msol), axis = false, 
        xlab = "", ylab = "", 
        frame = :none
    )    
    savefig("images/recover-inverse-flag.png")

    svdA = svd(A)
    plot(svdA.S, label = "singular values")
    plot!([1,length(svdA.S)],[10ϵ,10ϵ])
    savefig("images/svd-values-flag.png")

    inds = findall(svdA.S .> 10ϵ)

    Astable = svdA.U[:,inds] * Diagonal(svdA.S[inds]) * svdA.Vt[inds,:]
    norm(Astable - A) / norm(A)

    Ainv_stable = transpose(svdA.Vt[inds,:]) * Diagonal(1 ./ svdA.S[inds]) * transpose(svdA.U[:,inds])

    norm(Ainv_stable * A - I) / length(A)
    norm(A * Ainv_stable - I) / length(A)


    vsol = Ainv_stable * v
    norm(A * vsol - v) / norm(v)

    Msol = reshape(vsol, img_size...);

    plot(Gray.(Msol), axis = false, 
        xlab = "", ylab = "", 
        frame = :none
    )    
    savefig("images/recover-svd-flag.png")


    # Tikinov regulariser
    vsol = [A; sqrt(ϵ) * diagm(ones(Float64,size(A)[2]))] \ [v; zeros(size(A)[2])]
    
    # vsol = (A + sqrt(20ϵ) * I) \ v; 
    norm(A * vsol - v) / norm(v)

    Msol = reshape(vsol, img_size...);

    plot(Gray.(Msol), axis = false, 
        xlab = "", ylab = "", 
        frame = :none
    )    
    savefig("images/recover-tiki-flag.png")

    # plot(Gray.(Msol))
    # plot(Gray.(Mtrans))

## Masks example

    img = FileIO.load("images/masks.jpeg")
    gray_img = Gray.(img)
    M = Float64.(gray_img)

    i1, i2 = size(M)

    sample_rate = 20;
    M = M[1:sample_rate:i1,1:sample_rate:i2]

    # blur image 
    kernel(x,y) = exp(- x^2 / (2.2)^2 - y^2 / (2.2)^2)

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

    savefig("images/blur-masks.png")

    # Tikinov reg 

    vsol = [A; sqrt(ϵ) * diagm(ones(Float64,size(A)[2]))] \ [v; zeros(size(A)[2])]  
     
    norm(A * vsol - v) / norm(v)

    Msol = reshape(vsol, img_size...);

    plot(Gray.(Msol), axis = false, 
        xlab = "", ylab = "", 
        frame = :none
    )
    savefig("images/recover-tiki-masks.png")    

    # # SVD inversion

    # svdA = svd(A)

    # plot(svdA.S, label = "singular values")
    # plot!([1,length(svdA.S)],[7ϵ,7ϵ])

    # inds = findall(svdA.S .> 7ϵ)

    # Astable = svdA.U[:,inds] * Diagonal(svdA.S[inds]) * svdA.Vt[inds,:]
    # norm(Astable - A) / norm(A)

    # Ainv_stable = transpose(svdA.Vt[inds,:]) * Diagonal(1 ./ svdA.S[inds]) * transpose(svdA.U[:,inds])

    # norm(Ainv_stable * A - I) / length(A)
    # norm(A * Ainv_stable - I) / length(A)

    # vsol = Ainv_stable * v
    # norm(A * vsol - v) / norm(v)

    # Msol = reshape(vsol, img_size...);

    # # plot(Gray.(Mtrans))
    # # plot(Gray.(M))
    
    # plot(Gray.(Msol))
    # savefig("images/recover-svd-masks.png")


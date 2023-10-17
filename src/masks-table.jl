using Images
using FileIO

using Plots
using Statistics
using LinearAlgebra

    ## Masks example
    img = FileIO.load("images/masks.jpeg")
    gray_img = Gray.(img)
    M = Float64.(gray_img)

    i1, i2 = size(M)

    sample_rate = 15;
    sample_rate = 18;
    M = M[1:sample_rate:i1,1:sample_rate:i2]

    gr(size = (200 * 1.3, 200)) 
    plot(Gray.(M), axis = false, 
        xlab = "", ylab = "", 
        frame = :none
    ) 
    # savefig("images/small-masks.png")

    # blur image 
    kernel(x,y) = exp(- x^2 / (2.6)^2 - y^2 / (2.6)^2)

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
    # savefig("images/blur-total-flag.png")
    # savefig("images/blur-flag.png")


    # Tikinov regulariser
    h = 600
    gr(size = (1.3 * h,h))

    δ = ϵ
    δs = [5e-6, 1e-4, 2e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.4]
    ps = map(δs) do δ
        vsol = [A; sqrt(δ) * diagm(ones(Float64,size(A)[2]))] \ [v; zeros(size(A)[2])]
    
        # vsol = (A + sqrt(20ϵ) * I) \ v; 
        norm(A * vsol - v) / norm(v)

        Msol = reshape(vsol, img_size...);

        plot(Gray.(Msol), axis = false, 
            xlab = "", ylab = "", 
            frame = :none, title = "δ ≈ $(round(2e4δ)/(2e2))x 10^-2"
        )
    end

    # h = 500
    # gr(size = (1.3 * h,h))
    # h = 700 
    # gr(size = (1.6 * h,h))
    plot(ps...)
    savefig("images/tikh-table-masks.png")

    δs = [[1e-6,2e-6, 3e-6, 5e-6,1e-5,5e-5]; 0.001:0.002:0.015] |> collect
    errors = map(δs) do δ
        vsol = [A; sqrt(δ) * diagm(ones(Float64,size(A)[2]))] \ [v; zeros(size(A)[2])]
    
        # vsol = (A + sqrt(20ϵ) * I) \ v; 
        norm(A * vsol - v) 
    end

    using LaTeXStrings
    gr(size = (200 * 1.6,200 ))
    plot(δs, errors, linewidth = 2.0, 
        label = L"\|A x_\delta - y \|", 
        ylab = "", xlab = L"\delta"
    )
    
    error = norm(ϵ .* randn(length(M)))

    plot!([0.0,maximum(δs)],[error,error], lab = L"\varepsilon")
    savefig("images/tikh-delta.pdf")
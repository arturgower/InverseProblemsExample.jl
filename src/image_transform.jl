using Images
using FileIO
using ImageCore, TestImages
using Noise
using ImageFiltering
using FFTW
using Plots
using Statistics
using DSP
using PlutoUI
using LinearAlgebra


img = FileIO.load("img.png")
gray_img=Gray.(img)
M=Float64.(gray_img)


i, j = size(M)

s = 5;

M1 = M[1:5:i,1:5:j]

Gray.(M1)

function kernel(x,y, σ)
   
    ker=(1/(2π * σ))*exp(- x^2 / σ^2 - y^2 / σ^2)
   
    return ker
   
end


img_size=size(M1)

A=zeros(img_size[1],img_size[2],img_size[1],img_size[2])

for i in range img_size[1]
   
    for j in range img_size[2]
       
        for m in range img_size[1]
           
            for n in range img_size[2]
               
                A[i,j,m,n]=kernel(i-m,j-n,10)
               
            end
        end
    end
end

g=zeros(img_size[1],img_size[2])  


for i in range img_size[1]
   
    for j in range img_size[2]
       
        for m in range img_size[1]
           
            for n in range img_size[2]
               g[i,j]=g[i,j]+A[i,j,m,n]*M1[m,n]
               
               
            end
        end
    end
end
using Enzyme
using LinearAlgebra
Enzyme.API.printall!(true)
Enzyme.API.printtype!(true)

n = 10
function speelpenning(y, x)
    for i in 1:length(x)
        y[1] += x[i] * x[i]
    end
    return nothing
end

function reverse(y::VT, x::VT) where {VT}
    FT = eltype(x)
    rx = convert(VT, zeros(FT,n))
    ry = convert(VT, ones(FT,1))

    _x = Duplicated(x, rx)
    _y = Duplicated(y, ry)

    autodiff(speelpenning, _y, _x)
    return rx
end

function forward_over_reverse(y::VT, x::VT) where {VT}
    FT = eltype(x)
    dx = convert(VT, ones(FT,n)); rx = convert(VT, zeros(FT,n)); drx = convert(VT, zeros(FT,n))
    dy = convert(VT, zeros(FT,1)); ry = convert(VT, ones(FT,1)); dry = convert(VT, zeros(FT,1))

    function foo(y, x)
        autodiff_deferred(speelpenning, Const, y, x)
        return nothing
    end

    _x = Duplicated(Duplicated(x,rx), Duplicated(dx,drx))
    _y = Duplicated(Duplicated(y,ry), Duplicated(dy,dry))

    fwddiff(foo, _y, _x)
    return rx, drx
end
x = [i/(1.0+i) for i in 1:n]
# x = ones(n)
y = zeros(1)
speelpenning(y,x)
@show y

g = reverse(y, x)
g1, g2 = forward_over_reverse(y,x)
all(g .== g1)

# using CUDA

# cux = x |> CuArray
# cuy = y |> CuArray

# speelpenning(cuy,cux)

# cug = reverse(cuy, cux)
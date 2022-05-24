using Enzyme
using LinearAlgebra
Enzyme.API.printall!(true)
Enzyme.API.printtype!(true)

n = 2
function speelpenning(y, x)
    y .= x .* x
    return nothing
end

function reverse(y::VT, x::VT) where {VT}
    FT = eltype(x)
    rx = convert(VT, zeros(FT,n))
    ry = convert(VT, ones(FT,n))

    _x = Duplicated(x, rx)
    _y = Duplicated(y, ry)

    autodiff(speelpenning, _y, _x)
    return rx
end

function forward_over_reverse(y::VT, x::VT) where {VT}
    FT = eltype(x)
    dx = convert(VT, ones(FT,n)); rx = convert(VT, zeros(FT,n)); drx = convert(VT, zeros(FT,n))
    dy = convert(VT, zeros(FT,n)); ry = convert(VT, ones(FT,n)); dry = convert(VT, zeros(FT,n))

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
y = zeros(n)
speelpenning(y,x)

g = reverse(y, x)

# Crashes
g1, g2 = forward_over_reverse(y,x)
# This test should pass
@show all(g .== g1)
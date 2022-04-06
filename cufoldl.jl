using Enzyme
using CUDA

n = 10
function speelpenning(y, x)
    CUDA.@allowscalar y .= foldl(*, x; init=1.0)
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

x = [i/(1.0+i) for i in 1:n]
y = zeros(1)
speelpenning(y,x)

# g = reverse(y, x)

using CUDA

cux = x |> CuArray
cuy = y |> CuArray

speelpenning(cuy,cux)

cug = reverse(cuy, cux)
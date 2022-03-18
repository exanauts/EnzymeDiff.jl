using Enzyme

function speelpenning(y::AbstractVector{VT}, x::AbstractVector{VT}) where {VT}
    y[1] = reduce(*, x)
    return nothing
end

y = [0.0]
n = 10
x = [i/(1.0+i) for i in 1:n]
speelpenning(y,x)
println("Speelpenning(x): ", y)


# Reverse
rx = zeros(n)
ry = [1.0]
autodiff(speelpenning, Duplicated(y,ry), Duplicated(x,rx))
y = [0.0]
speelpenning(y,x)
@show rx
@show ry

rg = copy(rx)
errg = 0.0
for (i, v) in enumerate(x)
    global errg += abs(rx[i]-y[1]/v)
end

println("$(y[1]-1/(1.0+n)) error in function")
println("$errg error in gradient")

# Forward
dg = zeros(n)
dx = zeros(n)
dy = zeros(n)
for i in 1:n
    dx = zeros(n)
    dx[i] = 1
    dy = [0.0]
    fwddiff(speelpenning, Duplicated(y,dy), Duplicated(x,dx))
    dg[i] = dy[1]
    y = [0.0]
end
@show rg
@show dg
speelpenning(y,x)

# Forward over Reverse
function rspeelpenning(x, rx, y, ry)
    autodiff_deferred(speelpenning, Duplicated(y,ry), Duplicated(x,rx))
    return nothing
end

using Enzyme

n = 10
function speelpenning(y, x)
    y[1] = reduce(*, x)
    return nothing
end

x   = zeros(n)
dx  = zeros(n)
rx  = zeros(n)
drx = zeros(n)
y   = zeros(1)
dy  = zeros(1)
ry  = zeros(1)
dry = zeros(1)

x = Duplicated(Duplicated(x,dx), Duplicated(rx,drx))
y = Duplicated(Duplicated(y,dy), Duplicated(ry,dry))

function foo(x, y)
    fwddiff_deferred(speelpenning, x, y)
    return nothing
end

autodiff_deferred(foo, x, y)

using Enzyme

n = 10

x = [i/(1.0+i) for i in 1:n]
dx  = ones(n)
rx  = zeros(n); drx = zeros(n)
y   = zeros(1); dy  = zeros(1)
ry  = ones(1); dry = zeros(1)

function foo(x, y)
    # y .= mapreduce(identity,*, x; dims=:, init=1.0)
    y .= mapreduce(identity,*, x; dims=:)
    return nothing
end

function bar(x, y)
    autodiff_deferred(foo, Const, x, y)
    return nothing
end

_x = Duplicated(Duplicated(x,rx), Duplicated(dx,drx))
_y = Duplicated(Duplicated(y,ry), Duplicated(dy,dry))

fwddiff(bar, _x, _y)
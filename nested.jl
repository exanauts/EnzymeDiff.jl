using Enzyme

function foo(x)
    x .= x .* x 
    return nothing
end

function bar1(x)
    x .= x .* x
    return nothing
end

bar2(x) = bar1(x)

function bar(x)
    bar1(x)
    foo(x)
    bar2(x)
    return nothing
end

# joint reversal
dfoo(x) = autodiff(foo, Const, x)
dbar1(x) = autodiff(bar1, Const, x)
dbar2(x) = autodiff(bar2, Const, x)

stack = []

function dbar(x)
    push!(stack, copy(x.val))
    bar1(x.val)
    push!(stack, copy(x.val))
    foo(x.val)
    push!(stack, copy(x.val))
    bar2(x.val)

    # Reverse
    x.val .= pop!(stack)
    dbar2(x)
    x.val .= pop!(stack)
    dfoo(x)
    x.val .= pop!(stack)
    dbar1(x)
    return nothing
end

# Passive
x = [2.0]
bar(x)

# Split foo reversal
x = [2.0]
dx = [1.0]
autodiff(bar, Const, Duplicated(x,dx))
split = copy(dx)

# Joint foo reversal
x = [2.0]
dx = [1.0]
dbar(Duplicated(x,dx))
all(dx .== split)

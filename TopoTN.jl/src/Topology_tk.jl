###### SSH MODEL

function get_x_op_SSH_quantics(L,sites)
    f(x) = -2^(L-2)+div(x + 1, 2)
    mpo = get_diagonal_mpo(L, sites, f)/2^L
    return mpo
end


function get_sz_quantics(L,sites)
    f(x) = (-1)^(x + 1)
    mpo = get_diagonal_mpo(L, sites, f)
    return mpo
end


function get_SSH_hamiltonian(L, sites, t1, t2)
    function f(x)
        x % 2 == 0 ? t2 : t1
    end
    mpo = get_diagonal_mpo(L, sites, f)
    Ham = kinetic_1d_nn_custom(L, sites, mpo)
    ITensorMPS.truncate!(Ham;cutoff=1e-8)
    return Ham
end



function get_C_op_MPO_SSH(L,sites, t1, t2; Nchebychev = 300, maxdim = 15)
    Ham = get_SSH_hamiltonian(L, sites, t1, t2)
    factor = 5
    H = Ham/factor
    Tnlist = KPM_Tn(H,Nchebychev,sites,maxdim = maxdim)
    Id_op = MPO(sites, "Id")
    P = get_density_from_Tn(Tnlist,Nchebychev,fermi=0.0,maxdim = maxdim)
    x_op = get_x_op_SSH_quantics(L,sites)
    sz = get_sz_quantics(L,sites)
    Q = Id_op - P
    
    T1 = apply(P, apply(x_op, Q))
    T2 = apply(Q, apply(x_op, P))
    C_op = apply(sz, T1+T2)
    return C_op
end


function W1D(L,sites, t1, t2, i; Nchebychev = 200, maxbonddim = 15)
    C_op = get_C_op_MPO_SSH(L,sites, t1, t2, Nchebychev = Nchebychev, maxdim = maxbonddim)
    
    function Cmarker(x)
        psi1 = binary_to_MPS(Int(2*x-1), L, sites)
        psi2 = binary_to_MPS(Int(2*x), L, sites)
        C1 = inner(psi1,apply(C_op,psi1))
        C2 = inner(psi2,apply(C_op,psi2))
        return (C1 + C2)
    end

    return Cmarker(i)*2^L
end



function get_x_op_square(x_mid, L, sites, L_chain)
    f(x) = mod(x - 1, L_chain) - x_mid
    mpo = get_diagonal_mpo(L, sites, f)
    return mpo
end


function get_y_op_square(y_mid,L, sites, L_chain)
    f(x) =  div(x - 1, L_chain) - y_mid
    mpo = get_diagonal_mpo(L, sites, f)
    return mpo
end


function get_x_op_square_approx(x_mid, L, sites, L_chain,a)
    f(x) = mod(x - 1, L_chain) - x_mid
    g(x) = sin(f(x)/a)*a
    mpo = get_diagonal_mpo(L, sites, g)
    return mpo
end


function get_y_op_square_approx(y_mid,L, sites, L_chain, a)
    f(x) =  div(x - 1, L_chain) - y_mid
    g(x) = sin(f(x)/a)*a
    mpo = get_diagonal_mpo(L, sites, g)
    return mpo
end



# Naive implementation of the Chern operator
function get_C_op_MPO_from_P(P, L, sites, L_chain; maxbonddim = 15, type = Float64)
    P = ITensorMPS.truncate!(P; maxdim = maxbonddim, cutoff=1e-8)
    factor = 10
    Id_op = MPO(sites, "Id")
    print("Got density matrix!")
    Q = Id_op - P
    function f(rmid)
        rmid = Int(rmid)
        xmid = mod(rmid-1, L_chain) 
        ymid = div(rmid-1, L_chain)
        x_op = get_x_op_square(xmid,L,sites,L_chain)
        y_op = get_y_op_square(ymid,L,sites,L_chain)
        T1 = apply(Q, apply(x_op, apply(P, apply(y_op, Q, maxdim = maxbonddim, cutoff=1e-5)), maxdim = maxbonddim, cutoff=1e-5), maxdim = maxbonddim, cutoff=1e-5)
        T2 = apply(P, apply(x_op, apply(Q, apply(y_op, P, maxdim = maxbonddim, cutoff=1e-5)), maxdim = maxbonddim, cutoff=1e-5), maxdim = maxbonddim, cutoff=1e-5)
        C_op = 2im*pi*-(T1,T2,maxdim = maxbonddim)
        C_op = ITensorMPS.truncate!(C_op, cutoff = 1e-5)
        rmid_mps = binary_to_MPS(Int(rmid), L, sites)
        res = inner(rmid_mps, apply(C_op,rmid_mps))
        println(real(res))
        return real(res)
    end
    return f
end




# Quenched position operators
function get_sinx_op(L, sites, L_chain, a, xfunc)
    g(x) = sin(xfunc(x, L_chain) / a) * a
    return TopoTN.get_diagonal_mpo(L, sites, g)
end

function get_cosx_op(L, sites, L_chain, a, xfunc)
    g(x) = cos(xfunc(x, L_chain) / a) * a
    return TopoTN.get_diagonal_mpo(L, sites, g)
end

function get_siny_op(L, sites, L_chain, a, yfunc)
    g(x) = sin(yfunc(x, L_chain) / a) * a
    return TopoTN.get_diagonal_mpo(L, sites, g)
end

function get_cosy_op(L, sites, L_chain, a, yfunc)
    g(x) = cos(yfunc(x, L_chain) / a) * a
    return TopoTN.get_diagonal_mpo(L, sites, g)
end



# Returns a callable chern-number function. Made self-contained by taking `L` and `sites` explicitly.
function get_C_op_MPO_from_P_quenched(P, L, sites, xfunc, yfunc; l=nothing, Λ=10, maxdim=500)
    # If you pass l, we use it. Otherwise infer L_chain from L as before (L_chain = 2^l, num qubits = 2l -> L = 2l).
    if l === nothing
        l = div(L, 2)
    end
    L_chain = 2^l
    

    Q = MPO(sites, "Id") - P

    sinX_op = get_sinx_op(L, sites, L_chain, Λ, xfunc)
    cosX_op = get_cosx_op(L, sites, L_chain, Λ, xfunc)

    sinY_op = get_siny_op(L, sites, L_chain, Λ, yfunc)
    cosY_op = get_cosy_op(L, sites, L_chain, Λ, yfunc)

    sinY_P = apply(sinY_op, P; maxdim=maxdim)
    cosY_P = apply(cosY_op, P; maxdim=maxdim)

    P_sinX = apply(P, sinX_op; maxdim=maxdim)
    P_cosX = apply(P, cosX_op; maxdim=maxdim)

    sinY_Q = apply(sinY_op, Q; maxdim=maxdim)
    cosY_Q = apply(cosY_op, Q; maxdim=maxdim)

    Q_sinX = apply(Q, sinX_op; maxdim=maxdim)
    Q_cosX = apply(Q, cosX_op; maxdim=maxdim)

    println("op done")

    C1 = apply(P_sinX, Q; maxdim=maxdim)
    C1 = apply(C1, sinY_P; maxdim=maxdim)
    c1 = apply(Q_sinX, P; maxdim=maxdim)
    c1 = apply(c1, sinY_Q; maxdim=maxdim)
    C1 = c1 - C1
    println("C1 done")

    C2 = apply(P_cosX, Q; maxdim=maxdim)
    C2 = apply(C2, cosY_P; maxdim=maxdim)
    c2 = apply(Q_cosX, P; maxdim=maxdim)
    c2 = apply(c2, cosY_Q; maxdim=maxdim)
    C2 = c2 - C2
    println("C2 done")

    C3 = apply(P_sinX, Q; maxdim=maxdim)
    C3 = apply(C3, cosY_P; maxdim=maxdim)
    c3 = apply(Q_sinX, P; maxdim=maxdim)
    c3 = apply(c3, cosY_Q; maxdim=maxdim)
    C3 = c3 - C3
    println("C3 done")

    C4 = apply(P_cosX, Q; maxdim=maxdim)
    C4 = apply(C4, sinY_P; maxdim=maxdim)
    c4 = apply(Q_cosX, P; maxdim=maxdim)
    c4 = apply(c4, sinY_Q; maxdim=maxdim)
    C4 = c4 - C4
    println("C4 done")

    function calculate_chern_number(alpha)
        α = binary_to_MPS(alpha - 1, L, sites)
        x = xfunc(alpha - 1, L_chain)
        y = yfunc(alpha - 1, L_chain)
        ch =  cos(x/Λ) * cos(y/Λ) * inner(α', C1, α)
        ch += sin(x/Λ) * sin(y/Λ) * inner(α', C2, α)
        ch -= cos(x/Λ) * sin(y/Λ) * inner(α', C3, α)
        ch -= sin(x/Λ) * cos(y/Λ) * inner(α', C4, α)
        return ch * 2im * pi
    end

    return calculate_chern_number
end
# -*- coding: utf-8 -*-
using ITensors: MPO, MPS, OpSum, expect, inner, siteinds
using ITensors
using ITensorMPS
using LinearAlgebra
using Plots
using QuanticsTCI
import TensorCrossInterpolation as TCI
using TCIITensorConversion
using Quantics

ITensors.op(::OpName"sigma_plus",::SiteType"Qubit") =
 [0 1
  0 0]

ITensors.op(::OpName"sigma_minus",::SiteType"Qubit") =
 [0 0
  1 0]

#1. General tools
#####################################
#For the break when moving to next row

function break_chain(x_start, L_chain, num_site, sites)    
    xvals =  range(1, num_site; length=num_site)
    f(x) =  if isinteger((x_start-1 + x) / L_chain)
        0
    else
        1
    end 

    qtt, ranks, errors = quanticscrossinterpolate(Float64, f,  xvals; tolerance=1e-8)
    tt = TCI.tensortrain(qtt.tci)
    density_mps = ITensors.MPS(tt;sites)
    density_mpo = outer(density_mps',density_mps) 
    
    for i in 1:Int(log2(num_site))
        density_mpo.data[i] = Quantics._asdiagonal(density_mps.data[i],sites[i])
    end
    
    return  density_mpo 
end



#function break_chain(x_start, L_chain, num_site, sites)
#    L = Int(log2(num_site))
#    break_mpo = MPO(sites, "Id")
#    
#    inds = (x_start-1):L_chain:(num_site-1)
#    
    
    #THIS SCALES BADLY // TO CHECK
    #TO COMPARE WITH QUANTICS IMPLEMENTATION
#    projectors = map(n -> binary_to_MPS(n,L,sites), inds)
#    density_mps = reduce(+, projectors)

#    density_mpo = outer(density_mps',density_mps)
#    for i in 1:L
#        density_mpo.data[i] = Quantics._asdiagonal(density_mps.data[i],sites[i])
#    end
#    break_mpo = break_mpo - density_mpo
#    return break_mpo
#end


function break_chain_noQ(x_start, L_chain, num_site, sites)
    L = Int(log2(num_site))
    T = generate_kin_u(sites, num_site)

    for i in 1:Int(log2(L_chain))
        T = apply(T,T)
    end

    Id_op = MPO(sites, "Id")
    p = Id_op
    for i in 1:Int(log2(L_chain))
        p = apply(p, Id_op + T)
        T = apply(T,T)
    end

    mps1 = binary_to_MPS(L_chain-1, L ,sites)

    density_mps = mps1' * p
    break_mpo = MPO(sites, "Id")
    
    density_mpo = outer(density_mps',density_mps)
    
    for i in 1:L
        density_mpo.data[i] = Quantics._asdiagonal(density_mps.data[i],sites[i])
    end
    break_mpo = break_mpo - density_mpo
    return break_mpo
end


function generate_kin_u(sites, num_site)
    L = Int(log2(num_site))
    kinetic_1 = OpSum()
    for i in 1:L
        os = OpSum()
        os += 1,"sigma_plus",L-(i-1)

        for i in 1:L-i 
            os *=  ("Id",i) 
        end


        for i in L+2-i :L 
            os *=  ("sigma_minus",i) 
        end
        
        kinetic_1 += os
    end
    k_mpo_1 = MPO(kinetic_1,sites)
    return k_mpo_1
end

function generate_kin_d(sites, num_site)
    L = Int(log2(num_site))
    kinetic_2 = OpSum()
    for i in 1:L
        os = OpSum()
        os += 1,"sigma_minus",L-(i-1)

        for i in 1:L-i 
            os *=  ("Id",i) 
        end


        for i in L+2-i :L 
            os *=  ("sigma_plus",i) 
        end
        
        kinetic_2 += os
    end
 
    k_mpo_2 = MPO(kinetic_2,sites)
end

#all intra-row hopping
function intrachain_hopping(L_chain, num_site, sites; hopping = MPO(sites, "Id"), t = 1) 
    L = Int(log2(num_site))
    break_mpo = break_chain(L_chain, L_chain, num_site, sites)
    k_mpo_2 = generate_kin_d(sites, num_site)
    hop_2 = apply(hopping, k_mpo_2)
    true_hop_2 = apply(hop_2, break_mpo)

    k_mpo_1 = generate_kin_u(sites, num_site)
    hop_1 = apply(k_mpo_1,hopping)
    true_hop_1 = apply(break_mpo,hop_1)
    k_mpo =  +(t*true_hop_1, conj(t)*true_hop_2;  cutoff = 1e-8)
    return k_mpo
end



# shift the off-diag line
function arbitarty_offline(k_mpo,demand_order)
    k_mpo_o1 = k_mpo
    k_mpo_o2 = apply(k_mpo,k_mpo)
    target_mpo = k_mpo
    for iter_num in 1:demand_order
        if iter_num  == 1
            target_mpo = k_mpo_o1
        elseif iter_num  == 2
            target_mpo = k_mpo_o2
        else
            target_mpo = apply(k_mpo, k_mpo_o2)
            k_mpo_o2 = target_mpo
        end
    end
    return target_mpo
end



function to_binary_vector(n, size)
    # Convert to binary string
    binary_str = string(n, base=2)
    
    # Pad the binary string with leading zeros to match the desired size
    padded_binary_str = lpad(binary_str, size, '0')
    
    # Convert the padded string into a vector of strings (each character is a string)
    return collect(padded_binary_str) |> x -> map(s -> string(s), x)
end



#for validating of small system Hamiltonian
function get_matrix(mpo,size)
    mat = zeros(size, size) 
    for i in 0:size-1
        for j in 0:size-1
            element = round(inner(randomMPS(sites,to_binary_vector(Int(i),Int(log2(size))))',mpo,randomMPS(sites,to_binary_vector(Int(j),Int(log2(size))))))
            mat[i+1,j+1] = element
        end
    end
    return mat
end
#######################################




#2. For square lattice
################################
function interchain_hopping_square(L_chain, num_site, sites; hopping = MPO(sites, "Id"), t=1)
    L = Int(log2(num_site))
    k_mpo_1 = generate_kin_u(sites, num_site)
    hop_1 = apply(hopping,k_mpo_1)
    K_mpo_1_true = apply(hop_1,arbitarty_offline(k_mpo_1,L_chain-1))
        
    
    k_mpo_2 = generate_kin_d(sites, num_site)
    hop_2 = apply(k_mpo_2, hopping)
    K_mpo_2_true = apply(arbitarty_offline(k_mpo_2,L_chain-1),hop_2)
    k_mpo = t*K_mpo_1_true + conj(t)*K_mpo_2_true
    return k_mpo
end



##############################


function interchain_hopping_square_2nd_plus(L_chain, num_site, sites; hopping = MPO(sites, "Id"), t2 = 1)
    L = Int(log2(num_site))
    break_mpo = break_chain(L_chain, L_chain, num_site, sites)
    K_mpo_1 = generate_kin_u(sites, num_site)
    K_mpo_1_broken = apply(break_mpo,K_mpo_1)
    hop_1 = apply(hopping, K_mpo_1_broken)
    K_shift_1 = arbitarty_offline(K_mpo_1,L_chain +1-1)
    K_mpo_1_true = apply(K_shift_1, hop_1)
    
    K_mpo_2 = generate_kin_d(sites, num_site)
    K_mpo_2_broken = apply(K_mpo_2, break_mpo)
    hop_2 = apply(K_mpo_2_broken, hopping)
    K_shift_2 = arbitarty_offline(K_mpo_2,L_chain +1-1)
    K_mpo_2_true = apply(K_shift_2, hop_2)
    k_mpo = t2*K_mpo_1_true + conj(t2)*K_mpo_2_true
    return k_mpo
end


function interchain_hopping_square_2nd_minus(L_chain, num_site, sites; hopping = MPO(sites, "Id"), t2 = 1)
    L = Int(log2(num_site))
    break_mpo = break_chain(1, L_chain, num_site, sites)
    K_mpo_1 = generate_kin_u(sites, num_site)
    K_mpo_1_broken = apply(break_mpo,K_mpo_1)
    hop_1 = apply(hopping,K_mpo_1_broken)
    K_shift_1 = arbitarty_offline(K_mpo_1,L_chain -1-1)
    K_mpo_1_true = apply(hop_1,K_shift_1)
    
    K_mpo_2 = generate_kin_d(sites, num_site)
    K_mpo_2_broken = apply(K_mpo_2,break_mpo)
    hop_2 = apply(K_mpo_2_broken, hopping)
    K_shift_2 = arbitarty_offline(K_mpo_2,L_chain -1-1)
    K_mpo_2_true = apply(K_shift_2, hop_2)
    k_mpo = t2*K_mpo_1_true + conj(t2)*K_mpo_2_true
    return k_mpo
end

##############################



# 3. for triangular
# #############################
function skeleton(L_chain, num_site, sites)  
    xvals =  range(1, num_site; length=num_site)
    f(x) = if x % L_chain == 1
        return 0
    else
        return 1
    end

    qtt, ranks, errors = quanticscrossinterpolate(Float64, f,  xvals; tolerance=1e-8)
    tt = TCI.tensortrain(qtt.tci)
    density_mps = ITensors.MPS(tt;sites)
    density_mpo = outer(density_mps',density_mps)
    
    for i in 1:Int(log2(num_site))
        density_mpo.data[i] = Quantics._asdiagonal(density_mps.data[i],sites[i])
    end
 
    return density_mpo
end



function interchain_hopping_triangle(L_chain, num_site, sites) 
    L = Int(log2(num_site))
    kinetic_1 = OpSum()
    kinetic_2 = OpSum()
    for i in 1:L
        os = OpSum()
        os += 1,"sigma_plus",L-(i-1)

        for i in 1:L-i 
            os *=  ("Id",i) 
        end


        for i in L+2-i :L 
            os *=  ("sigma_minus",i) 
        end
        
        kinetic_1 += os
    end
    k_mpo_1 = MPO(kinetic_1,sites)
    tri_hop_1 = skeleton(L_chain, num_site, sites) 
   
    
    K_mpo_1_ture = arbitarty_offline(k_mpo_1,L_chain)
    tri_connect_1 = arbitarty_offline(k_mpo_1,L_chain-1)
    tri_hop_1_ture = apply(tri_hop_1,tri_connect_1)
   
    for i in 1:L
        os = OpSum()
        os += 1,"sigma_minus",L-(i-1)

        for i in 1:L-i 
            os *=  ("Id",i) 
        end


        for i in L+2-i :L 
            os *=  ("sigma_plus",i) 
        end
        
        kinetic_2 += os
    end
 
    k_mpo_2 = MPO(kinetic_2,sites)
    K_mpo_2_ture = arbitarty_offline(k_mpo_2,L_chain)
    tri_connect_2 = arbitarty_offline(k_mpo_2,L_chain-1)


    tri_hop_2_ture = apply(tri_connect_2,tri_hop_1)
    k_mpo =  K_mpo_2_ture + K_mpo_1_ture+ tri_hop_2_ture+ tri_hop_1_ture

    return k_mpo
end
##############################

# 4. honeycomb
##############################
# function odd_template_old(L_chain, num_site, sites)    

#     xvals =  range(1, num_site; length=num_site)
#     f(x) = if x % 2 == 1
#         1
#     else
#         0
#     end

#     qtt, ranks, errors = quanticscrossinterpolate(Float64, f,  xvals; tolerance=1e-8)
#     tt = TCI.tensortrain(qtt.tci)
#     density_mps = ITensors.MPS(tt;sites)
#     density_mpo = outer(density_mps',density_mps)
    
#     for i in 1:Int(log2(num_site))
#         density_mpo.data[i] = Quantics._asdiagonal(density_mps.data[i],sites[i])
#     end
    
#     return density_mpo
# end

# # +
 
# function even_template_old(L_chain, num_site, sites)    
#     xvals =  range(1, num_site; length=num_site)
#      f(x)= if (x % 2 == 0) ||((x-1) % (L_chain) == 0)
#         0
 
#     else
#         1
#     end

#     qtt, ranks, errors = quanticscrossinterpolate(Float64, f,  xvals; tolerance=1e-8)
#     tt = TCI.tensortrain(qtt.tci)
#     density_mps = ITensors.MPS(tt;sites)
#     density_mpo = outer(density_mps',density_mps) 
    
#     for i in 1:Int(log2(num_site))
#         density_mpo.data[i] = Quantics._asdiagonal(density_mps.data[i],sites[i])
#     end
    
#     return  density_mpo 
# end
 
# # -

# function odd_skeleton_old(L_chain, num_site, sites)    

#     xvals =  range(0, num_site-1; length=num_site)
#     f(x) = if floor(Int, x / (L_chain)) % 2 == 0
#         return 1
#     else
#         return 0
#     end

#     qtt, ranks, errors = quanticscrossinterpolate(Float64, f,  xvals; tolerance=1e-8)
#     tt = TCI.tensortrain(qtt.tci)
#     density_mps = ITensors.MPS(tt;sites)
#     density_mpo = outer(density_mps',density_mps) 
    
#     for i in 1:Int(log2(num_site))
#         density_mpo.data[i] = Quantics._asdiagonal(density_mps.data[i],sites[i])
#     end
    
#     return   density_mpo 
# end

# function even_skeleton_old(L_chain, num_site, sites)    

#     xvals =  range(0, num_site-1 ; length=num_site)
#     f(x) = if floor(Int, (x ) / (L_chain )) % 2 == 0
#         return 0
#     else
#         return 1
#     end

#     qtt, ranks, errors = quanticscrossinterpolate(Float64, f,  xvals; tolerance=1e-8)
#     tt = TCI.tensortrain(qtt.tci)
#     density_mps = ITensors.MPS(tt;sites)
#     density_mpo = outer(density_mps',density_mps) 
    
#     for i in 1:Int(log2(num_site))
#         density_mpo.data[i] = Quantics._asdiagonal(density_mps.data[i],sites[i])
#     end
 
#     return   density_mpo
# end

function odd_template(L_chain, num_site, sites)    
    f(x) = Float64(isodd(x))
    L = Int(log2(num_site))
    mpo = get_diagonal_mpo(L, sites, f)
    return mpo
end
# +
 
function even_template(L_chain, num_site, sites)    
    f(x) = Float64(!(x % 2 == 0 || (x - 1) % L_chain == 0))
    L = Int(log2(num_site))
    mpo = get_diagonal_mpo(L, sites, f)
    return  mpo 
end
 
# -

function odd_skeleton(L_chain, num_site, sites)    
    f(x) = Float64(iseven(div(x,L_chain)))
    L = Int(log2(num_site))
    mpo = get_diagonal_mpo(L,sites, f)
    return   mpo 
end

function even_skeleton(L_chain, num_site, sites)    
    f(x) = Float64(isodd(div(x,L_chain)))
    L = Int(log2(num_site))
    mpo = get_diagonal_mpo(L, sites, f)
    return   mpo
end

function interchain_hopping_honeycomb(L_chain, num_site, sites) 
    L = Int(log2(num_site))

    odd_hop_tp = odd_template(L_chain, num_site, sites)
    even_hop_tp = even_template(L_chain, num_site, sites) 
    odd_hop_sk = odd_skeleton(L_chain, num_site, sites)
    even_hop_sk = even_skeleton(L_chain, num_site, sites)
    
    connect_mpo_up = apply(odd_hop_tp,odd_hop_sk)
    connect_mpo_dn = apply(even_hop_tp,even_hop_sk)
 
    kinetic_1 = OpSum()
    kinetic_2 = OpSum()
    for i in 1:L
        os = OpSum()
        os += 1,"sigma_plus",L-(i-1)

        for i in 1:L-i 
            os *=  ("Id",i) 
        end


        for i in L+2-i :L 
            os *=  ("sigma_minus",i) 
        end
        
        kinetic_1 += os
    end
    k_mpo_1 = MPO(kinetic_1,sites)
 
   
        
    K_mpo_1_up = arbitarty_offline(k_mpo_1,L_chain+1)
    K_mpo_1_dn = arbitarty_offline(k_mpo_1,L_chain-1)
     
    true_hop_up_1 = apply(connect_mpo_up, K_mpo_1_up)
    true_hop_dn_1 = apply(connect_mpo_dn, K_mpo_1_dn)
    
    for i in 1:L
        os = OpSum()
        os += 1,"sigma_minus",L-(i-1)

        for i in 1:L-i 
            os *=  ("Id",i) 
        end


        for i in L+2-i :L 
            os *=  ("sigma_plus",i) 
        end
        
        kinetic_2 += os
    end
 
    k_mpo_2 = MPO(kinetic_2,sites)
    K_mpo_2_up = arbitarty_offline(k_mpo_2,L_chain+1)
    K_mpo_2_dn = arbitarty_offline(k_mpo_2,L_chain-1)
     

    true_hop_up_2 = apply( K_mpo_2_up,connect_mpo_up)
    true_hop_dn_2 = apply( K_mpo_2_dn, connect_mpo_dn)
    k_mpo =  true_hop_up_1 + true_hop_up_2 + true_hop_dn_1 + true_hop_dn_2 

    return k_mpo
end



 

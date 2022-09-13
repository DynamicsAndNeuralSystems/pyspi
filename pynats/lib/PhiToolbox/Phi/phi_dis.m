function phi = phi_dis(type_of_phi, Z, probs)

N_st = probs.number_of_states;
q_probs = est_q(Z,N_st,probs);
phi = phi_dis_pq(type_of_phi, probs, q_probs);

end
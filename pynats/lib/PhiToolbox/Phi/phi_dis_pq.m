function phi = phi_dis_pq(type_of_phi, probs, q_probs)

switch type_of_phi
    case 'MI1'
        phi = MI1_dis(probs,q_probs);
    case 'star'
        phi = phi_star_dis(probs,q_probs);
    case 'SI'
        phi = SI_dis(probs,q_probs);
end

end
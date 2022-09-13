function x_t = generate_Boltzmann(beta,W,N,T)

x_t = zeros(N,T);

for t=1: T
    if t > 1
        input = beta*W*x_t(:,t-1);
        prob_vec = sigmoid(input);
    else
        prob_vec = rand(N,1);
    end
    logic_vec = rand(N,1) < prob_vec;
    
    for i=1: N
        if logic_vec(i)
            x_t(i,t) = 1;
        else
            x_t(i,t) = -1;
        end
    end
    
end

x_t = ((x_t -1)/2 + 1) + 1;

end


function y = sigmoid(x)
    y = 1./ (1 + exp(-x));
end
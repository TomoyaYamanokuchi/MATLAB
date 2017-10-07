function R = norm_colomun(S)



R1 = diag(S'*S);
R2 = R1;
R3 = -2*(S'*S);
R_2p = (R1 + R3)' + R2; 
R = sqrt(R_2p);
end
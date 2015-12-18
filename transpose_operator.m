%Compute the p,q commutation matrix T, such that
%T*vec(X) = vec(X') for p times q matrices X
%Martin Sundin, 2015-08-26

function T = transpose_operator(p,q)

T = sparse(p*q,p*q);
for i = 1:p
    for j = 1:q
        T(j+(i-1)*q,i+(j-1)*p) = 1;
    end
end

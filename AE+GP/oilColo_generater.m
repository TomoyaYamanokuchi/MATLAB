function l = oilColo_generater(h)


n = size(h, 1);
l = zeros(n, 3);

for i=1:n
   d = h(i,:);
   find(d)
   if find(d)==1
       l(i,:) = [1 0 0];
   elseif find(d) == 2
       l(i,:) = [0 1 0];
   else
       l(i,:) = [0 0 1];
   end
end

% a(a==1) = 'r';
% a(a==2) = 'b';
% a(a==3) = 'g';


end
function array = label_get_oilFlow(X)

array = zeros(size(X, 1), 1);

for i=1:size(X, 1)
   temp = X(i,:);
   index = find(temp==1);
   if index == 1
       array(i)='r';
   elseif index == 2
       array(i)='g';
   else
       array(i) = 'b';
   end
end


end
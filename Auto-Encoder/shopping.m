function result = shopping()

apple_unit_price = 100;
apple_num = 2;
orange_unit_price = 150;
orange_num = 3;
tax = 1.1;

% layer
mul_apple_layer = MulLayer();
mul_orange_layer = MulLayer();
add_apple_orange_layer = AddLayer();
mul_tax_layer = MulLayer();

% forward
apple_total_price = mul_apple_layer.forward(apple_unit_price, apple_num);
orange_total_price = mul_orange_layer.forward(orange_unit_price, orange_num);
total_price_excluding_tax = add_apple_orange_layer.forward(apple_total_price, orange_total_price);
total_price = mul_tax_layer.forward(total_price_excluding_tax, tax);

% backward
dtotal_price = 1;
[dtotal_price_excluding_tax, dtax] = mul_tax_layer.backward(dtotal_price);
[dapple_total_price, dorange_total_price] = add_apple_orange_layer.backward(dtotal_price_excluding_tax);
[dapple_unit_price, dapple_num] = mul_apple_layer.backward(dapple_total_price);
[dorange_unit_price, dorange_num] = mul_orange_layer.backward(dorange_total_price);

disp(total_price);
result = {dapple_num dapple_unit_price dorange_unit_price dorange_num dtax};


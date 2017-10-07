function [dapple_price, dtax] = go(apple, apple_num, dprice)
mul_layer_apple = MulLayer();
apple_price = mul_layer_apple.forward(apple, apple_num);
mul_layer_tax = MulLayer();
[dapple_price, dtax] = mul_layer_tax.backward(dprice);
end
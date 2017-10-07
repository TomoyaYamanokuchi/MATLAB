function stop = autoencoder_images(L)
stop = false;

% -- 1 -----------------------------
subplot(3, 2, 2);
imagesc(reshape(Y(1, :), 28, 28)*255);
axis equal;
subplot(3, 2, 1);
imagesc(reshape(X(1, :), 28, 28)*255);
axis equal;
% -- 6 -----------------------------
subplot(3, 2, 4);
imagesc(reshape(Y(2, :), 28, 28)*255);
axis equal;
subplot(3, 2, 3);
imagesc(reshape(X(2, :), 28, 28)*255);
axis equal;
% -- 8 -----------------------------
subplot(3, 2, 6);
imagesc(reshape(Y(3, :), 28, 28)*255);
axis equal;
subplot(3, 2, 5);
imagesc(reshape(X(3, :), 28, 28)*255);
axis equal;



end
function [centroid] = spectral_centroid(file)
[y, Fs] = audioread(file);

N = 2^12;                     % number of points to analyze
c = fft(y(1:N));            % compute fft of sound data
p = 2 * abs( c(2:N/2));         % compute power at each frequency
f = (1:N/2-1)*Fs/N;           % frequency corresponding to p
display(f);
sc = (Fs/N) * (sum(p .* (1:N/2-1))) ./ sum(p);
centroid = sc;
end

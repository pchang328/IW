function test_fourier(hi, noise)
if hi 
    dir_name = 'Sounds/High/';
    if noise
        txt_file = 'Data/spec_centroids_hi_noise.txt';
    else
        txt_file = 'Data/spec_centroids_hi_final_new.txt';
    end 
    
else   
    dir_name = 'Sounds/Low/';
    if noise
        txt_file = 'Data/spec_centroids_lo_noise.txt';
    else
        txt_file = 'Data/spec_centroids_lo_final_new.txt';
    end
end

format_string = '%s %e \n';
fileID = fopen(txt_file,'w');

high_sound_files = dir(strcat(dir_name,'*.wav'));
num_high_files = length(high_sound_files);
for i = 1:num_high_files
    file_name = high_sound_files(i).name;
    file_path = strcat(dir_name,file_name);
    [y, Fs] = audioread(file_path);
    if noise
        result = noise_ratio(y, Fs);
    else
        result = spectral_centroid(y, Fs);
    end
    fprintf(fileID, format_string, file_name, result);
end
end

function [centroid] = spectral_centroid(y, Fs)
    N = 2^12;                     % number of points to analyze
    c = fft(y(1:N));            % compute fft of sound data
    p = 2 * abs( c(2:N/2));         % compute power at each frequency
    % f = (1:N/2-1)*Fs/N;           % frequency corresponding to p
    sc = (Fs/N) * (sum(p .* (1:N/2-1))) ./ sum(p);
    centroid = sc;
end

function [noise] = noise_ratio(y, Fs)
    N = 2^12;                     % number of points to analyze
    c = fft(y(1:N))/N;            % compute fft of sound data
    p = 2*abs( c(2:N/2));         % compute power at each frequency
    f = (1:N/2-1)*Fs/N;           % frequency corresponding to p
    
    % Making sure you only take the frequencies <= 4000
    for i =  1:length(f)
        if f(i) > 4000 
            cut_off = i - 1;
            break
        end    
    end

    % Initialzing to new values
    f_new = f(1:cut_off);
    p_new = p(1:cut_off);
    total_energy = sum(p_new);
    num_max = 5;

    for i = 1:num_max
        [argvalue, argmax] = max(p_new);
        % These are the points that you want to delete +/- k %.
        args_to_remove_below = (argmax - 9):argmax;
        if argmax <= 9 
            args_to_remove_below = 1:argmax; 
        end
        args_to_remove_above = (argmax + 1):(argmax + 9);
        args_to_remove = horzcat(args_to_remove_below, args_to_remove_above);
        p_new(args_to_remove) = [];
        f_new(args_to_remove) = [];
    end
    total_noise_energy = sum(p_new);
    noise = total_noise_energy / total_energy;
    display(noise)
end


% function [centroid] = spectral_centroid(file)
% 
% [y, Fs] = audioread(file);
% X = fft(y);
% T=abs(X);
% lAbs = T(:, 2); % left channel DFT
% N = length(lAbs);
% lAbs = lAbs(1:N/2);
% N_new = length(lAbs);
% 
% 
% % left channel Spectral centroid
% X = zeros(N_new, 1);
% for i = 1:N_new
%     X(i) = i*lAbs(i);
% end
% 
% coef = Fs / N_new;
% centroid = coef * ((sum(X)) ./ sum(lAbs));
% 
% end





function plot_power_spec (file, remove)
    [y, Fs] = audioread(file);
    t = (1:length(y))/Fs;         % time

    ind = find(t>0.1 & t<0.12);   % set time duration for waveform plot
    figure; subplot(1,2,1)
    plot(t(ind),y(ind))  
    axis tight         
    title(['Waveform of ' file])

    N = 2^12;                     % number of points to analyze
    c = fft(y(1:N))/N;            % compute fft of sound data
    p = 2*abs( c(2:N/2));         % compute power at each frequency
    f = (1:N/2-1)*Fs/N;           % frequency corresponding to p

    if remove 
        % Let us only consider the frequences upto 4000
        for i =  1:length(f)
            if f(i) > 4000 
                cut_off = i - 1;
                break
            end    
        end

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
        display(total_noise_energy / total_energy);
        f = f_new;
        p = p_new;
    end

    subplot(1,2,2)
    semilogy(f, p)
    axis([0 4000 10^-4 1])                
    title(['Power Spectrum of ' file])

end



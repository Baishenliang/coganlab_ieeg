clc; clear; close all;

% Set the data path
data_path = 'C:\Users\bl314\Box\CoganLab\task_stimuli\LexicalDecRepDelay';
output_file = 'envelope_power_bins.txt';

% Get all WAV files
wav_files = dir(fullfile(data_path, '*.wav'));
num_files = length(wav_files);

% Optimized frequency bands for speech signals (in Hz)
freq_bands = [50 400; 400 1000; 1000 3000; 3000 5000; 5000 8000];

% Initialize storage variables
power_values = zeros(num_files, 5);
file_names = cell(num_files, 1);

% Process each file
for i = 1:num_files
    % Read the audio file
    file_name = wav_files(i).name;
    disp(['Now doing ', file_name])
    [audio, fs] = audioread(fullfile(data_path, file_name));

    % Convert stereo to mono if necessary
    if size(audio, 2) > 1
        audio = mean(audio, 2);
    end

    % Compute envelope power for each frequency band
    for j = 1:size(freq_bands, 1)
        % Design a bandpass filter for the current frequency band
        bpFilt = designfilt('bandpassiir', 'FilterOrder', 4, ...
            'HalfPowerFrequency1', freq_bands(j, 1), ...
            'HalfPowerFrequency2', freq_bands(j, 2), ...
            'SampleRate', fs);
        
        % Apply zero-phase filtering to avoid phase distortion
        filtered_audio = filtfilt(bpFilt, audio);
        
        % Extract the temporal envelope using the Hilbert transform
        envelope = abs(hilbert(filtered_audio));
        
        % Compute the mean envelope power
        power_values(i, j) = mean(envelope .^ 2);
    end

    % Store the file name without the ".wav" extension
    file_names{i} = erase(file_name, '.wav');
end

% Normalize power values to the range [0,1]
min_vals = min(power_values, [], 1);
max_vals = max(power_values, [], 1);
normalized_power_1 = (power_values - min_vals) ./ (max_vals - min_vals);

% Log transform
log_transformed_powers = log(normalized_power_1 + 0.01); 

% Normalize again
normalized_power = (log_transformed_powers - min(log_transformed_powers)) ./ (max(log_transformed_powers) - min(log_transformed_powers));

% Prepare output data
output_data = [file_names, num2cell(normalized_power)];

% Plot the distributions:
figure;
hist(normalized_power(:,1));
figure;
hist(normalized_power(:,2));
figure;
hist(normalized_power(:,3));
figure;
hist(normalized_power(:,4));
figure;
hist(normalized_power(:,5));

% Write results to a text file
fid = fopen(output_file, 'w');
for i = 1:num_files
    fprintf(fid, '%s\t', output_data{i, 1});
    fprintf(fid, '%.6f\t', output_data{i, 2:end});
    fprintf(fid, '\n');
end
fclose(fid);

disp('Envelope power computation completed. Results saved to envelope_power_optimized.txt');

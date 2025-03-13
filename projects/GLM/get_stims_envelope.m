% Clone the IoSR-Surrey toolbox
% https://github.com/IoSR-Surrey/MatlabToolbox.git:
addpath(genpath('../../../MatlabToolbox'))
% iosr.install % run it for the first time

clc; clear; close all;

% Set the data path
data_path = 'C:\Users\bl314\Box\CoganLab\task_stimuli\LexicalDecRepDelay';
output_file = 'envelope_power_bins.txt';

% Get all WAV files
wav_files = dir(fullfile(data_path, '*.wav'));
num_files = length(wav_files);

% Optimized frequency bands (using gamma tone) for speech signals (in Hz)
MakeErbCFs=@iosr.auditory.makeErbCFs;
gammatoneFast=@iosr.auditory.gammatoneFast;
Fmin = 50;	% lower frequency of filterbank in Hz
Fmax = 8e3;	% upper frequency of filterbank (.8 * Nyquist)
bandnum =  16 ; % THE NUMBER OF BANDS
cfs = MakeErbCFs(Fmin,Fmax,bandnum);

% print freqBands
freqBands = cell(1,bandnum);
for k = 1:bandnum
    freqBands{k} = sprintf('%.2f Hz', cfs(k));
end
disp(freqBands);

% Initialize storage variables
power_values = zeros(num_files, bandnum);
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
    [~,env,~] = gammatoneFast(audio'/std(audio),cfs,fs);

    % Compute the mean envelope power
    power_values(i, :) = mean(env .^ 2,2);

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
for j=1:bandnum
    figure;
    hist(normalized_power(:,j));
end

% Write results to a text file
fid = fopen(output_file, 'w');
for i = 1:num_files
    fprintf(fid, '%s\t', output_data{i, 1});
    fprintf(fid, '%.6f\t', output_data{i, 2:end});
    fprintf(fid, '\n');
end
fclose(fid);

disp('Envelope power computation completed. Results saved to envelope_power_optimized.txt');

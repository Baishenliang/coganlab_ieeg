% Get the user's home directory dynamically
homeDir = getenv('USERPROFILE'); % Works on Unix-based systems

% Define the folder path relative to the home directory
folderPath = fullfile(homeDir, 'Box', 'CoganLab', 'task_stimuli', 'LexicalDecRepDelay');

% Get all .wav files in the folder
wavFiles = dir(fullfile(folderPath, '*.wav'));

% Initialize an array to store durations of audio files
durations = [];

% Loop through each .wav file
for i = 1:length(wavFiles)
    disp(i)
    % Get the full path of the file
    filePath = fullfile(folderPath, wavFiles(i).name);
    
    % Read the audio data and sampling rate
    [audioData, sampleRate] = audioread(filePath);
    
    % Calculate the duration of the audio in seconds
    duration = length(audioData) / sampleRate;
    
    % Store the duration in the array
    durations = [durations; duration];
end

% Calculate the mean duration of all audio files
meanDuration = mean(durations);

% Plot a histogram of the audio durations
figure;
histogram(durations, 10); % Use 10 bins for the histogram; adjust as needed
title('Histogram of Audio Durations');
xlabel('Duration (seconds)');
ylabel('Frequency');

% Print the mean duration in the console
fprintf('The mean duration of the audio files is %.2f seconds.\n', meanDuration);

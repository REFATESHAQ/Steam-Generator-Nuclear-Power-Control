%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Project Name: Modeling and Control of a Steam Generator in a Nuclear Power Plant: Ensuring Performance and Safety.
% Written By: Refat Mohammed Abdullah Eshaq, on October 19, 2024.
% Copyright (c) 2024, Refat Mohammed Abdullah Eshaq, All rights reserved.
% This code is licensed under a GNU Affero General Public License Version 3 (GNU AGPLv3), for more information, see <https://www.gnu.org/licenses/agpl-3.0.en.html>.
% This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
% This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details. You should have received a copy of the GNU Affero General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
% This work has been supported by my livelihood and my family's aid. 
% The code and data will be  connected to my futuer work
% Publish Date: December 19, 2024.  
% Author's Email: refateshaq1993@gmail.com; refateshaq@hotmail.com; fs18050005@cumt.edu.cn; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%% Code starts from here.
clc;
clear;
close all;
rng(0);
% Load data from the file
Datacurve; % Load data (PowerUTSG, FeedWaterUTSG, steamUTSG, FeedWaterTemp, IncomWaterTemp)



% Extract time and data from columns
time_Q = PowerUTSG(:, 1);  % Time from column 1 of PowerUTSG
Q = PowerUTSG(:, 2) / 1000; % Convert MW to GW

time_Wfw = FeedWaterUTSG(:, 1);   % Time from column 1 of FeedWaterUTSG
Wfw = FeedWaterUTSG(:, 2) / 1000;  % Convert Kg/s to Mg/s

time_Wst = steamUTSG(:, 1);        % Time from column 1 of steamUTSG
Wst = steamUTSG(:, 2) / 1000;      % Convert Kg/s to Mg/s

time_Tfw = FeedWaterTemp(:, 1);    % Time from column 1 of FeedWaterTemp
Tfw = FeedWaterTemp(:, 2);         % Temperature of secondary coolant 

time_Tsg = IncomWaterTemp(:, 1);   % Time from column 1 of IncomWaterTemp
Tsg = IncomWaterTemp(:, 2);        % Temperature of primary coolant

time_mea = Measured(:, 1);         % Time from column 1 of IncomWaterTemp
Ld_measured = Measured(:, 2);      % Temperature of primary coolant

% Ensure unique time points
[time_Q, unique_idx_Q] = unique(time_Q); Q = Q(unique_idx_Q);
[time_Wfw, unique_idx_Wfw] = unique(time_Wfw); Wfw = Wfw(unique_idx_Wfw);
[time_Wst, unique_idx_Wst] = unique(time_Wst); Wst = Wst(unique_idx_Wst);
[time_Tfw, unique_idx_Tfw] = unique(time_Tfw); Tfw = Tfw(unique_idx_Tfw);
[time_Tsg, unique_idx_Tsg] = unique(time_Tsg); Tsg = Tsg(unique_idx_Tsg);
[time_mea, unique_idx_Tsg] = unique(time_mea); Ld_measured = Ld_measured(unique_idx_Tsg);

% Define new time vector
new_time = (0:0.6:600)';

Q_interp = interp1(time_Q, Q, new_time, 'linear', 'extrap');
Wfw_interp = interp1(time_Wfw, Wfw, new_time, 'linear', 'extrap');
Wst_interp = interp1(time_Wst, Wst, new_time, 'linear', 'extrap');
Tfw_interp = interp1(time_Tfw, Tfw, new_time, 'linear', 'extrap');
Tsg_interp = interp1(time_Tsg, Tsg, new_time, 'linear', 'extrap');
Ld_interp = interp1(time_mea, Ld_measured, new_time, 'linear', 'extrap');

% Combine data into a matrix
clean_data = [Q_interp, Wfw_interp, Wst_interp, Tsg_interp, Tfw_interp]; % Each is a column vector of size 1001x1

% Set noise parameters
noise_params = struct( ...
    'gaussian', struct('level', 0.05), ...
    'impulse', struct('prob', 0.1, 'strength', 0.5), ...
    'salt_pepper', struct('prob', 0.05), ...
    'uniform', struct('level', 0.03), ...
    'periodic', struct('freq', 0.1, 'amplitude', 0.02), ...
    'sinusoidal', struct('freq', 0.05, 'amplitude', 0.05), ...
    'random_walk', struct('step_size', 0.01), ...
    'temperature', struct('amplitude', 0.02, 'freq', 0.01), ...
    'vibration', struct('amplitude', 0.05, 'freq', 0.06), ...
    'pink', struct('power', -1), ...
    'brownian', struct('power', -2), ...
    'blue', struct('power', 1), ...
    'violet', struct('power', 2), ...
    'thermal', struct('level', 0.1), ...
    'shot', struct('rate', 0.5), ...
    'flicker', struct('power', -0.5), ...
    'white', struct('level', 0.02), ...
    'quantization', struct('step_size', 0.01), ...
    'burst', struct('prob', 0.01, 'amplitude', 0.5), ...
    'modulated', struct('freq', 0.1, 'amplitude', 0.03), ...
    'subtractive', struct('prob', 0.05), ...
    'intermodulation', struct('freq1', 0.2, 'freq2', 0.3, 'amplitude', 0.03), ...
    'phase', struct('deviation', 0.01), ...
    'aliased', struct('freq', 0.4, 'sampling_rate', 0.1), ...
    'jitter', struct('amount', 0.001), ...
    'environmental', struct('level', 0.02), ...
    'oscillatory', struct('freq', 0.05, 'amplitude', 0.03) ...
);

% Specify the types of noise you want to add
selected_noises = {'oscillatory','gaussian'}; % Modify this list as needed

% Add selected noise to the clean data
noisy_data = add_selected_noise(clean_data, noise_params, selected_noises);

% Function to add specified types of noise
function noisy_data = add_selected_noise(data, noise_params, selected_noises)
    noisy_data = data; % Start with the clean data
    
    % Loop through each selected noise type and add it to the data
    for i = 1:length(selected_noises)
        noise_type = selected_noises{i};
        
        switch noise_type
            case 'gaussian'
                gaussian_noise = noise_params.gaussian.level * randn(size(data));
                noisy_data = noisy_data + gaussian_noise;
                
            case 'impulse'
                impulse_noise = zeros(size(data));
                spike_indices = rand(size(data)) < noise_params.impulse.prob;
                impulse_noise(spike_indices) = noise_params.impulse.strength * randn(sum(spike_indices(:)), 1);
                noisy_data = noisy_data + impulse_noise;
                
            case 'salt_pepper'
                sp_noise = zeros(size(data));
                sp_indices = rand(size(data)) < noise_params.salt_pepper.prob;
                sp_noise(sp_indices) = 2 * (rand(sum(sp_indices(:)), 1) > 0.5) - 1;
                noisy_data = noisy_data + sp_noise;
                
            case 'uniform'
                uniform_noise = noise_params.uniform.level * (2 * rand(size(data)) - 1);
                noisy_data = noisy_data + uniform_noise;
                
            case 'periodic'
                periodic_noise = noise_params.periodic.amplitude * sin(2 * pi * noise_params.periodic.freq * (1:size(data, 1))');
                noisy_data = noisy_data + periodic_noise;
                
            case 'sinusoidal'
                sinusoidal_noise = noise_params.sinusoidal.amplitude * sin(2 * pi * noise_params.sinusoidal.freq * (1:size(data, 1))');
                noisy_data = noisy_data + sinusoidal_noise;
                
            case 'random_walk'
                random_walk_noise = cumsum(noise_params.random_walk.step_size * randn(size(data)), 1);
                noisy_data = noisy_data + random_walk_noise;
                
            case 'temperature'
                temp_noise = noise_params.temperature.amplitude * sin(2 * pi * noise_params.temperature.freq * (1:size(data, 1))');
                noisy_data = noisy_data + temp_noise;
                
            case 'vibration'
                vib_noise = noise_params.vibration.amplitude * sin(2 * pi * noise_params.vibration.freq * (1:size(data, 1))');
                noisy_data = noisy_data + vib_noise;
                
            case 'pink'
                pink_noise = noise_params.pink.power * (1./fft(randn(size(data))));
                noisy_data = noisy_data + pink_noise;
                
            case 'brownian'
                brown_noise = cumsum(randn(size(data)), 1) * noise_params.brownian.power;
                noisy_data = noisy_data + brown_noise;
                
            case 'blue'
                blue_noise = noise_params.blue.power * (randn(size(data)) .* (1:size(data,1))');
                noisy_data = noisy_data + blue_noise;
                
            case 'violet'
                violet_noise = noise_params.violet.power * (randn(size(data)) .* (1:size(data,1))'.^2);
                noisy_data = noisy_data + violet_noise;
                
            case 'thermal'
                thermal_noise = noise_params.thermal.level * randn(size(data));
                noisy_data = noisy_data + thermal_noise;
                
            case 'shot'
                shot_noise = poissrnd(noise_params.shot.rate, size(data));
                noisy_data = noisy_data + shot_noise;
                
            case 'flicker'
                flicker_noise = noise_params.flicker.power * (1./fft(randn(size(data))));
                noisy_data = noisy_data + flicker_noise;
                
            case 'white'
                white_noise = noise_params.white.level * randn(size(data));
                noisy_data = noisy_data + white_noise;
                
            case 'quantization'
                quantization_noise = round(data / noise_params.quantization.step_size) * noise_params.quantization.step_size - data;
                noisy_data = noisy_data + quantization_noise;
                
            case 'burst'
                burst_noise = zeros(size(data));
                burst_indices = rand(size(data)) < noise_params.burst.prob;
                burst_noise(burst_indices) = noise_params.burst.amplitude * randn(sum(burst_indices(:)), 1);
                noisy_data = noisy_data + burst_noise;
                
            case 'modulated'
                modulated_noise = noise_params.modulated.amplitude * sin(2 * pi * noise_params.modulated.freq * (1:size(data, 1))') .* randn(size(data));
                noisy_data = noisy_data + modulated_noise;
                
            case 'subtractive'
                subtractive_noise = zeros(size(data));
                sub_indices = rand(size(data)) < noise_params.subtractive.prob;
                subtractive_noise(sub_indices) = -data(sub_indices);
                noisy_data = noisy_data + subtractive_noise;
                
            case 'intermodulation'
                intermod_noise = noise_params.intermodulation.amplitude * sin(2 * pi * (noise_params.intermodulation.freq1 + noise_params.intermodulation.freq2) * (1:size(data, 1))');
                noisy_data = noisy_data + intermod_noise;
                
            case 'phase'
                phase_noise = noise_params.phase.deviation * (1:size(data, 1))';
                noisy_data = noisy_data + phase_noise .* data;
                
            case 'aliased'
                aliased_noise = noise_params.aliased.freq * sin(2 * pi * (1:size(data, 1))' / noise_params.aliased.sampling_rate);
                noisy_data = noisy_data + aliased_noise;
                
            case 'jitter'
                jitter_noise = noise_params.jitter.amount * randn(size(data));
                noisy_data = noisy_data + jitter_noise;
                
            case 'environmental'
                environmental_noise = noise_params.environmental.level * randn(size(data));
                noisy_data = noisy_data + environmental_noise;
                
            case 'oscillatory'
                oscillatory_noise = noise_params.oscillatory.amplitude * sin(2 * pi * noise_params.oscillatory.freq * (1:size(data, 1))');
                noisy_data = noisy_data + oscillatory_noise;
                
            otherwise
                warning('Unknown noise type: %s', noise_type);
        end
    end
end

add_noisee = false;

if add_noisee
Q_interp = noisy_data(:,1);
Wfw_interp = noisy_data(:,2);
Wst_interp = noisy_data(:,3);
Tfw_interp = noisy_data(:,4);
Tsg_interp = noisy_data(:,5);
else
Q_interp = Q_interp ;
Wfw_interp = Wfw_interp;
Wst_interp = Wst_interp;
Tfw_interp = Tfw_interp;
Tsg_interp = Tsg_interp;
end

power_decreas = true;

if power_decreas

Q_interp = Q_interp;
Wfw_interp = Wfw_interp;
Wst_interp = Wst_interp;
Tfw_interp = Tfw_interp;
Tsg_interp = Tsg_interp;

% Initial conditions
Ld0 = 11.23;
xM0 = 0.329;
p0 = 6890000;

else 
Q_interp = flipud(Q_interp);
Wfw_interp = flipud(Wfw_interp);
Wst_interp = flipud(Wst_interp);
Tfw_interp = flipud(Tfw_interp);
Tsg_interp = flipud(Tsg_interp);

% Initial conditions
Ld0 = 11;
xM0 = 0.063;
p0 = 9340000;

end 


% Declare global variables for derivatives
global dLd_dt_log dxM_dt_log dp_dt_log t_log;

% Initialize empty arrays to store derivatives
dLd_dt_log = [];
dxM_dt_log = [];
dp_dt_log = [];
t_log = [];


% Define ODE functions
function dydt = steam_generator_ode(t, y, Wst_interp, Wfw_interp, Q_interp, Tsg_interp, Tfw_interp, new_time)
    global dLd_dt_log dxM_dt_log dp_dt_log t_log;

    Ld  = y(1); % Liquid level
    xM  = y(2); % Dryness fraction 
    p   =  y(3); 
    %Interpolate inputs
    Wst = interp1(new_time, Wst_interp, t, 'linear', 'extrap');
    Wfw = interp1(new_time, Wfw_interp, t, 'linear', 'extrap');
    Q   = interp1(new_time, Q_interp, t, 'linear', 'extrap');
    Tsg = interp1(new_time, Tsg_interp, t, 'linear', 'extrap');
    Tfw = interp1(new_time, Tfw_interp, t, 'linear', 'extrap');

 % ODEs
    dLd_dt = (1 / 10) * (-0.8 * (1 + Ld / 10) + (1 / xM) * Wst + 0.23 * (1 + 1 / xM) * Wfw);
    dxM_dt = (1 / 10) * ((1 + Ld / 10) * (-0.2 * xM - 3.3 * xM^2) + (0.14 + 3 * xM) * Q);

Aq = 8.3; 
Aw = 5.74;
Ad = 9.9;
k1 = 0.65;
k =1; 
kd = 15.923;
L2 = 14.8;
Vst = 41.9;
vd = (1.5e-6 * (Tfw+500) + 0.001164958062);
rho_d = 1/vd;
rho_g = ((5.9e-6 * p) + 25.4);
vg = 1 ./ rho_g;
rho_f = ((2.79e-5 * p) + 814);
vf = 1 ./ rho_f;
d_rho_g_dp = 5.9*10^-5;

  term1 = ((Ld / vd)) - ((Aq / Aw) * ((1 - (k1 * xM)) / (vf + (k1 * xM * vg)))) * L2;
    Wd = (kd * sqrt(term1));
    Wsep = (Ad * rho_d * dLd_dt + Wd - k * Wfw) / (1 - xM);  
    dp_dt = (xM * Wsep - Wst) / (Vst * d_rho_g_dp);

  % Append current time and derivatives to logs
    dLd_dt_log = [dLd_dt_log; dLd_dt];
    dxM_dt_log = [dxM_dt_log; dxM_dt];
    dp_dt_log = [dp_dt_log; dp_dt];
    t_log = [t_log; t];
    
    dydt = [dLd_dt; dxM_dt;dp_dt];
end



% Solve the ODE system
[t, y] = ode45(@(t, y) steam_generator_ode(t, y, Wst_interp, Wfw_interp, Q_interp, Tsg_interp, Tfw_interp, new_time), new_time, [Ld0, xM0,p0]);

% Extract solution for Ld, xM, p
Ld = y(:, 1);
xM = y(:, 2);
p  = y(:, 3);

% Remove duplicate time points in t_log and corresponding values in the logs
[t_log_unique, unique_indices] = unique(t_log);  % Find unique time points and their indices
dLd_dt_log_unique = dLd_dt_log(unique_indices);  % Use indices to get corresponding unique values
dxM_dt_log_unique = dxM_dt_log(unique_indices);
dp_dt_log_unique = dp_dt_log(unique_indices);

% Now perform interpolation with unique time points
dLd_dt = interp1(t_log_unique, dLd_dt_log_unique, t, 'linear', 'extrap');
dxM_dt = interp1(t_log_unique, dxM_dt_log_unique, t, 'linear', 'extrap');
dp_dt = interp1(t_log_unique, dp_dt_log_unique, t, 'linear', 'extrap');

Aq = 8.3; 
Aw = 5.74;
Ad = 9.9;
k1 = 0.65;
k =1; 
kd = 15.923;
Vst = 41.9;

vd = (1.5e-6 .* (Tfw_interp+273) + 4.7*10^-4);
rho_d = 1./vd;
rho_g = ((5.9e-6 .* p) + 25.4);
vg = 1 ./ rho_g;
rho_f = ((2.79e-5 .* p) + 814);
vf = 1 ./ rho_f;

%vd = 1/858.5;
%vf = 0.0012285;
%vg = 0.03877149877

vs = vf + xM .* vg;



term11 = (((Ld .*rho_d)) - ((Aq / Aw) .* ((1 - 0.65 .* xM) ./ ( vf+ 0.65 .* xM .* vg))) .* 14.8);
Wd1 = 15.923 .* sqrt(term11); 
Wsep = (Ad .* rho_d .*dLd_dt + Wd1 - k .* Wfw_interp) ./ (1 - xM); 
Lw1 = (((Aq/Aw)) .* ((vf./vs)) .* (1-k1.*xM)).*14.8;

%Lw2 = vf .* ((Ld./vd)-(Wd1.^2 ./ kd.^2));


figure;
plot(t, Wd1, 'b-', 'LineWidth', 2);
hold on; 
plot(t, Wsep, 'r-', 'LineWidth', 2);
hold off; 

figure;  
plot(t,  Lw1, 'r-', 'LineWidth', 2);
figure;
plot(t, p, 'r-', 'LineWidth', 2);
%figure;  
%plot(t,  Lw2, 'k-', 'LineWidth', 2);


figure;
% Plot Liquid level (Ld) over time
subplot(2, 1, 1);
plot(t, Ld, 'b-', 'LineWidth', 2);
title('Liquid Level (Ld) over Time');
xlabel('Time (s)');
ylabel('L_d (m)');
grid on;

% Plot Steam Quality (xM) over time
subplot(2, 1, 2);
plot(t, xM, 'b-', 'LineWidth', 2);
title('Steam Quality (xM) over Time');
xlabel('Time (s)');
ylabel('x_M');
grid on;


figure;
% Plot dLd/dt
subplot(3, 1, 1);
plot(t_log, dLd_dt_log, 'b');
xlabel('Time (s)');
ylabel('dL_d/dt (m/s)');
title('Velocity (dL_d/dt) over Time');

% Plot dxM/dt
subplot(3, 1, 2);
plot(t_log, dxM_dt_log, 'r');
xlabel('Time (s)');
ylabel('dx_M/dt');
title('Rate of Change of Steam Quality (dx_M/dt) over Time');

% Plot dp/dt
subplot(3, 1, 3);
plot(t_log, dp_dt_log, 'g');
xlabel('Time (s)');
ylabel('dp/dt');
title('Rate of Change of Pressure (dp/dt) over Time');



% Assuming Ldl is the water level data from the simulation
Ldl_shifted = Ld - Ld0;  % Shift so that initial value is zero


% Save the interpolated data
save('data.mat', 'Wfw_interp', 'Wst_interp', 'Tfw_interp', 'Tsg_interp', 'Q_interp' );

%%%####################################################################################%%%

% Define matrices A and B
A = [-0.008, -0.4157;
    -0.0009, -0.2537];
 
B = [0.3183, 0;
     0, 0.1083];

% Define initial conditions for the states
x0 = [0; 0.329];  % [delta_Ld; delta_xM] initial values

% Define the inputs (interpolated input functions if time-dependent)
% Ensure time_input is correctly assigned
time_input = new_time;

delta_Q = Q_interp;
delta_Wst =  Wst_interp;
delta_Wfw = Wfw_interp;

% Interpolated functions for inputs
u_Q = @(t) interp1(time_input, delta_Q, t, 'linear', 'extrap');
u_Wst = @(t) interp1(time_input, delta_Wst, t, 'linear', 'extrap');
u_Wfw = @(t) interp1(time_input, delta_Wfw, t, 'linear', 'extrap');

% Define the system of differential equations
function dx = linear_system(t, x, A, B, u_Q, u_Wst, u_Wfw)
    % Get the interpolated input values at time t
    delta_Q = u_Q(t);
    delta_Wst = u_Wst(t);
    delta_Wfw = u_Wfw(t);
    
    % Construct the input vector
    u = [delta_Wst; delta_Q];
    
    % Calculate the state derivative
    dx = A * x + B * u + [0.0303 * delta_Wfw; 0];
end

% Set up the time span for simulation
tspan = [0 600];

% Run the simulation
[T, X] = ode45(@(t, x) linear_system(t, x, A, B, u_Q, u_Wst, u_Wfw), tspan, x0);

% Plot the results
figure;
plot(t, Ldl_shifted, 'b','DisplayName', 'L_{d} (Nonlinear model)', 'LineWidth', 2);
hold on
plot(T, X(:, 1), 'r', 'DisplayName', 'L_{d} (Linear model)', 'LineWidth', 2);
plot(t,Ld_interp, 'k--', 'DisplayName', 'L_{d} (Measured data)', 'LineWidth', 2);
hold off
title('Water leve variation  L_d over Time');
xlabel('Time (s)');
ylabel('Water leve variation (L_{d})');
grid on;
legend show;

figure;
plot(t, xM, 'b','DisplayName','x_{M} (Nonlinear model)', 'LineWidth', 2);
hold on
plot(T, X(:, 2), 'r','DisplayName','x_{M} (Linear model)', 'LineWidth', 2);
title('Steam Quality (x_{M}) over Time');
xlabel('Time (s)');
ylabel('x_M');
grid on;
legend show;









 
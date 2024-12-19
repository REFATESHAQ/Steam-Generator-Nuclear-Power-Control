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
clear;
clc;
close all;

load('data.mat');

%Q_interp = Q_noise;
%Wfw_interp = Wfw_noise;
%Wst_interp = Wst_noise;


new_time = (0:0.6:600)';  % Time vector
num_steps = length(new_time);

% Initialize state variables and control inputs
Ld = zeros(num_steps, 1);  % Liquid level (output)
xM = zeros(num_steps, 1);  % Steam quality (output)
Q_control = Q_interp;  % Heat source (input)
Wfw_control = Wfw_interp;  % Feedwater flow (input)
Wst_control = Wst_interp;  % Steam flow (input)

% Initial conditions
Ld(1) = 11.34;  % Initial liquid level
xM(1) = 0.3142;  % Initial steam quality

% NMPC Parameters
predictionHorizon = 10;  % Prediction horizon (in terms of time steps)
dt = 0.6;  % Time step for the simulation

% Desired values for Ld and xM
reference_Ld = 12;  % Desired liquid level in meters
reference_xM = 0.3142;  % Desired steam quality

% Control input bounds
lb = [0.16 * ones(predictionHorizon, 1);   % Lower bound for Q
      0.01 * ones(predictionHorizon, 1);   % Lower bound for Wfw
      0.01 * ones(predictionHorizon, 1)];  % Lower bound for Wst

ub = [0.2 * ones(predictionHorizon, 1);   % Upper bound for Q (set to 2 GW)
      0.2 * ones(predictionHorizon, 1);  % Upper bound for Wfw (set to 100 Mg/s)
      0.2 * ones(predictionHorizon, 1)];  % Upper bound for Wst (set to 100 Mg/s)

% Optimization options for fmincon
options = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp');


disturbance_scale = 0;  % Adjust as needed


%% Loop over time steps
for i = 1:num_steps-1
    % Define initial guesses for the control inputs (Q, Wfw, Wst)
    Q_init = Q_control(i) * ones(predictionHorizon, 1);
    Wfw_init = Wfw_control(i) * ones(predictionHorizon, 1);
    Wst_init = Wst_control(i) * ones(predictionHorizon, 1);
    
    % Pack initial guess into one vector
    u_init = [Q_init; Wfw_init; Wst_init];
    
    % Define cost function
    cost_function = @(u) nmpc_cost_function(u, Ld(i), xM(i), predictionHorizon, dt, reference_Ld, reference_xM);
    
    % Define nonlinear constraints (system dynamics)
    nonlinear_constraints = @(u) nmpc_constraints(u, Ld(i), xM(i), predictionHorizon, dt);
    
    % Solve the optimization problem using fmincon
    [u_opt, ~] = fmincon(cost_function, u_init, [], [], [], [], lb, ub, nonlinear_constraints, options);
    
    % Extract the first control actions
    Q_control(i) = u_opt(1);
    Wfw_control(i) = u_opt(predictionHorizon + 1);
    Wst_control(i) = u_opt(2 * predictionHorizon + 1);
    
    % Simulate the system using the control inputs
    disturbance = disturbance_scale * randn;
    
    dLd_dt = (1/10) * (-0.8 * (1 + Ld(i)/10) + (1/xM(i)) * Wst_control(i) + 0.23 * (1 + 1/xM(i)) * Wfw_control(i));
    dxM_dt = (1/10) * ((1 + Ld(i)/10) * (-0.2 * xM(i) - 3.3 * xM(i)^2) + (0.14 + 3 * xM(i)) * Q_control(i));
  
    dLd_dt = dLd_dt + disturbance;
    dxM_dt = dxM_dt + disturbance;
    
    % Update the states using forward Euler integration
    Ld(i + 1) = max(min(Ld(i) + dt * dLd_dt, 15), 5);
    xM(i + 1) = max(min(xM(i) + dt * dxM_dt, 0.5), 0.1);
end

% Plot results
figure;

% Plot Liquid level (Ld) over time
subplot(3, 1, 1);
plot(new_time, Ld, 'b-', 'LineWidth', 2);
title('Liquid Level (Ld) over Time');
xlabel('Time (s)');
ylabel('L_d (m)');
grid on;

% Plot Steam quality (xM) over time
subplot(3, 1, 2);
plot(new_time, xM, 'r-', 'LineWidth', 2);
title('Steam Quality (xM) over Time');
xlabel('Time (s)');
ylabel('x_M');
grid on;

% Plot control inputs over time
subplot(3, 1, 3);
plot(new_time, Q_control, 'g-', 'LineWidth', 2); hold on;
plot(new_time, Wfw_control, 'b--', 'LineWidth', 2);
plot(new_time, Wst_control, 'r-.', 'LineWidth', 2);
title('Control Inputs over Time');
xlabel('Time (s)');
ylabel('Control Inputs');
legend('Q (GW)', 'Wfw (Mg/s)', 'Wst (Mg/s)');
grid on;

% Adjust layout for better spacing
sgtitle('NMPC Control and System Response');

%% Cost function for NMPC
function J = nmpc_cost_function(u, Ld, xM, predictionHorizon, dt, reference_Ld, reference_xM)
    Q = u(1:predictionHorizon);
    Wfw = u(predictionHorizon + 1:2 * predictionHorizon);
    Wst = u(2 * predictionHorizon + 1:end);
    
    % Initialize cost
    J = 0;
    
    % Simulation loop for cost calculation
    for k = 1:predictionHorizon
        % System dynamics (discretized)
        dLd_dt = (1/10) * (-0.8 * (1 + Ld/10) + (1/xM) * Wst(k) + 0.23 * (1 + 1/xM) * Wfw(k));
        dxM_dt = (1/10) * ((1 + Ld/10) * (-0.2 * xM - 3.3 * xM^2) + (0.14 + 3 * xM) * Q(k));
        
        % Update states
        Ld = Ld + dt * dLd_dt;
        xM = xM + dt * dxM_dt;
        
        % Cost based on tracking error
        error = [Ld - reference_Ld; xM - reference_xM];
        J = J + error' * error;  % Cost for tracking error (simple L2 norm)
    end
    
    % Penalize control effort (optional)
    J = J + 0.1 * (sum(Q.^2) + sum(Wfw.^2) + sum(Wst.^2));
end

%% Nonlinear constraints for NMPC (dynamics)
function [c, ceq] = nmpc_constraints(u, Ld, xM, predictionHorizon, dt)
    Q = u(1:predictionHorizon);
    Wfw = u(predictionHorizon + 1:2 * predictionHorizon);
    Wst = u(2 * predictionHorizon + 1:end);
    
    % Initialize equality constraints (system dynamics)
    ceq = [];
    
    % Initialize inequality constraints with default values
    c = zeros(predictionHorizon * 7, 1);  % 7 constraints per time step

    % Loop through prediction horizon to check states
    for k = 1:predictionHorizon
        % System dynamics (discretized)
        dLd_dt = (1/10) * (-0.8 * (1 + Ld/10) + (1/xM) * Wst(k) + 0.23 * (1 + 1/xM) * Wfw(k));
        dxM_dt = (1/10) * ((1 + Ld/10) * (-0.2 * xM - 3.3 * xM^2) + (0.14 + 3 * xM) * Q(k));
        
        % Update states for constraints
        Ld_next = Ld + dt * dLd_dt;
        xM_next = xM + dt * dxM_dt;
        
        % Populate `c` with constraint values for each time step `k`
        idx = (k-1)*7 + 1;  % Starting index for constraints at time step k
        
        % State constraints
        c(idx)   = Ld_next - 15;    % Max limit for Ld
        c(idx+1) = 5 - Ld_next;     % Min limit for Ld
        c(idx+2) = xM_next - 0.5;   % Max limit for xM
        c(idx+3) = 0.1 - xM_next;   % Min limit for xM

        % Control input constraints
        c(idx+4) = Wfw(k) - 100;    % Max limit for Wfw
        c(idx+5) = Wst(k) - 100;    % Max limit for Wst
        c(idx+6) = Q(k) - 2;        % Max limit for Q
    end
    
    % No equality constraints
    ceq = [];  % Keep ceq empty if no equality constraints are needed
end





clear; clc; close all;

scale_factor_W = 1e4;
scale_factor_CRB = 100;
num_samples = 57;

save_digits = 3;
solve_fail = 0;

Nt = 12;
Nr = 10;

K = 7;

filename = sprintf('ISAC_Dataset_Wk_%d_RL.mat', K);
if exist(filename, 'file')
    fprintf('An existing file was detected: %s\n', filename);
    fprintf('Loading old data...\n');
    old_data = load(filename);
    
    old_features = old_data.final_features;
    old_labels = old_data.final_labels;
    
    fprintf('Number of samples in the old file: %d\n', size(old_features, 1));
else
    fprintf('File not found %s，Create new file。\n', filename);
    old_features = [];
    old_labels = [];
end

feat_dim = 1 + 1 + 1 + Nt*K + Nt*K;

m = Nt * (Nt - 1) / 2;

real_len_per_user = Nt + m;
imag_len_per_user = m;

compact_len_per_user = real_len_per_user + imag_len_per_user;
W_dim = compact_len_per_user * K;

label_dim = W_dim + 1;

data_features = zeros(num_samples, feat_dim);
data_labels = zeros(num_samples, label_dim);
valid_indices = false(num_samples, 1);

L = 10;

sigma2C_dBm = 0;
sigma2R_dBm = 0;

sigma2C = 10^((sigma2C_dBm - 30)/10);
sigma2R = 10^((sigma2R_dBm - 30)/10);

Gamma_dB = 10;
Gamma_lin = 10^(Gamma_dB/10);
Gamma_vec = Gamma_lin * ones(K,1);

lambda = 1;
d = lambda/2;
k0 = 2*pi/lambda;

n_t = (-((Nt-1)/2):((Nt-1)/2)).';
n_r = (-((Nr-1)/2):((Nr-1)/2)).';

for loop_idx = 1:num_samples
    PT_dBm_min = 12;
    PT_dBm_max = 13;
    PT_dBm = PT_dBm_min + (PT_dBm_max - PT_dBm_min) * rand();
    PT = 10^((PT_dBm - 30)/10);

    H = (randn(K, Nt) + 1j * randn(K, Nt)) / sqrt(2);
    Z_C = sqrt(sigma2C/2) * (randn(K, Nt) + 1j * randn(K, Nt));
    H = H + Z_C;

    theta_deg_min = 20;
    theta_deg_max = 30;
    theta_deg = theta_deg_min + (theta_deg_max - theta_deg_min) * rand();
    theta = deg2rad(theta_deg);
    a_theta = exp(1j * k0 * d * n_t * sin(theta));
    b_theta = exp(1j * k0 * d * n_r * sin(theta));
    G = alpha * (b_theta * a_theta');
    A = b_theta * a_theta';
    a_dot = 1j * k0 * d * n_t * cos(theta) .* a_theta;
    b_dot = 1j * k0 * d * n_r * cos(theta) .* b_theta;
    Ad = b_dot * a_theta' + b_theta * a_dot';

    Q = zeros(Nt, Nt, K);
    for k = 1:K
        hk = H(k,:).';
        Q(:,:,k) = hk * hk';
    end

    cvx_begin quiet
        cvx_precision best 
    
        variable W(Nt, Nt, K) hermitian semidefinite
        variable t 
        
        expression RX(Nt, Nt)
        RX = zeros(Nt, Nt);
        for k = 1:K
            RX = RX + W(:,:,k);
        end
        
        a11 = real(trace(Ad' * Ad * RX));
        a12 = trace(Ad' * A  * RX);
        a22 = real(trace(A'  * A  * RX));
        
        LMI = [a11 - t,   a12;
               conj(a12), a22];
        
        LMI == hermitian_semidefinite(2); 
        
        for k = 1:K
            Gammak = Gamma_vec(k);
            Qk = Q(:,:,k);
            
            signal_k = trace(Qk * W(:,:,k));
            
            interference_k = 0;
            for i = 1:K
                if i ~= k
                    interference_k = interference_k + trace(Qk * W(:,:,i));
                end
            end
            interference_k = Gammak * interference_k;
            
            noise_k = Gammak * sigma2C;
            
            real(signal_k - interference_k) >= noise_k;
        end
        
        total_power = 0;
        for k = 1:K
            total_power = total_power + trace(W(:,:,k));
        end
        real(total_power) <= PT;
        
        minimize(-t)
    cvx_end
    
    if strcmp(cvx_status, 'Solved')
        valid_indices(loop_idx) = true;

        W_opt = full(W);
        RX_opt = full(RX);

        num_CRB = sigma2R * trace(A' * A * RX_opt);
        term1 = trace(Ad' * Ad * RX_opt);
        term2 = trace(A'  * A  * RX_opt);
        term3 = trace(Ad' * A  * RX_opt);
        
        den_CRB = 2 * abs(alpha)^2 * L * ( term1 * term2 - abs(term3)^2 );
        CRB_rad = real(num_CRB / den_CRB);
        
        CRB_deg  = CRB_rad * (180/pi)^2;
        CRB_deg2   = sqrt(CRB_deg);

        H_feat = zeros(1, 2 * Nt * K);
        for k = 1:K
            base = (k-1) * 2 * Nt;
            H_feat(base + (1:Nt)) = real(H(k, :));
            H_feat(base + (Nt+1 : 2*Nt)) = imag(H(k, :));
        end
       
        feature_vector = [theta, PT_dBm, Gamma_lin, H_feat];
        
        W_scaled = W_opt * scale_factor_W;
        CRB_scaled = CRB_deg2 * scale_factor_CRB;

        W_compact_all = zeros(1, compact_len_per_user * K);

        for k = 1:K
            Wk = W_scaled(:,:,k);

            Wk = (Wk + Wk') / 2;

            [real_part, imag_part] = encode_hermitian_compact(Wk);
            user_block = [real_part, imag_part];
            
            idx = (k-1) * compact_len_per_user + (1:compact_len_per_user);
            W_compact_all(idx) = user_block;
        end

        label_vector = [W_compact_all, CRB_scaled];

        feature_vector = round(feature_vector, save_digits);
        label_vector   = round(label_vector, save_digits);

        data_features(loop_idx, :) = feature_vector;
        data_labels(loop_idx, :) = label_vector;

        if mod(loop_idx, 50) == 0
            fprintf('Loop %d: OK. CRB=%.4f\n', loop_idx, CRB_deg2);
        end

    else
        solve_fail = solve_fail + 1;
        valid_indices(loop_idx) = false;
        if mod(loop_idx, 10) == 0
            fprintf('Loop %d/%d: Failed (%s)\n', loop_idx, num_samples, cvx_status);
        end
    end
end

current_valid_features = data_features(valid_indices, :);
current_valid_labels = data_labels(valid_indices, :);

fprintf('\nThe dataset has been generated.\nNumber of valid samples:%d\nNumber of failed samples:%d\n', ...
    size(current_valid_features, 1), solve_fail);

final_features = [old_features; current_valid_features];
final_labels   = [old_labels; current_valid_labels];
fprintf('Total sample size after merging: %d\n', size(final_features, 1));

save(filename, ...
    'final_features', 'final_labels', 'save_digits', ...
    'scale_factor_W', 'scale_factor_CRB', ...
    'Nt', 'Nr', 'K', 'PT_dBm_min', 'PT_dBm_max', ...
    'theta_deg_min', 'theta_deg_max', ...
    'real_len_per_user', 'imag_len_per_user', 'compact_len_per_user'); 

fprintf('The data has been saved as: %s\n', filename);

TEMP = load(filename)

function [real_part, imag_part] = encode_hermitian_compact(Wk)
    Nt = size(Wk, 1);

    diag_r = real(diag(Wk)).';

    mask = triu(true(Nt), 1);
    upper = Wk(mask);

    upper_re = real(upper).';
    upper_im = imag(upper).';

    real_part = [diag_r, upper_re];
    imag_part = upper_im;
end

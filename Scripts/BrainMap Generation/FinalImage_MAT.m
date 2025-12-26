%% FinalImage_MAT.m 
% Render ULM-style MatOut images directly from the PINN track export produced by this repository.
%
% Attribution / provenance
% -----------------------
% This script is an *adapted framework* based on the same ULM toolbox conventions used by
% the original authors of `ULM_Track2MatOut.m` (Arthur Chavignon et al., Team PPM).
% The goal is to make the output of this repo (Track_tot_* tracks) directly consumable by
% `ULM_Track2MatOut` so that MatOut volumes and images can be generated from exported tracks.
%
% `ULM_Track2MatOut.m` is distributed under Creative Commons BY-NC-SA 4.0 (see its header).
% Please cite the original ULM toolbox references when using their rendering functions.
%
% Minimal inputs (only these are required)
% ---------------------------------------
%   1) ulmToolboxDir : folder containing `ULM_Track2MatOut.m`
%   2) trackMatFile  : this repo's exported tracks .mat (Track_tot_1/2/3)
%   3) outputDir     : folder where PNGs and a *_matouts.mat will be saved
%
% Notes
% -----
% - Tracks exported by this repo are stored as [x, z, vx, vz, timeline].
% - ULM_Track2MatOut expects per-track arrays with columns:
%     [z_index, x_index, vz, vx]
% - This script does NOT require IQ/UF files: it builds the render grid from the
%   track bounding box.
% - If you know frameRateHz and lambda_mm, you can optionally convert velocity to mm/s.

%% ===================== USER CONFIG =====================
ulmToolboxDir = '/ABS/PATH/TO/ULM_toolbox';        % must contain ULM_Track2MatOut.m
trackMatFile  = '/ABS/PATH/TO/*_PINN_tracks.mat';  % contains Track_tot_1/2/3
outputDir     = '/ABS/PATH/TO/output';
res = 10;                 % SR grid scale: SRscale = 1/res (track-units per SR pixel)
padding_pixels = 20;      % padding around bounding box (in SR pixels)

% Plot orientation
if ~exist('swapXZ','var') || isempty(swapXZ)
    swapXZ = true;        % if true: swap x/z axes in figures (helps match desired view)
end

% Velocity magnitude colormap scaling (percentiles of nonzero velocity values)
% Tip: if your map looks "all blue", increase the low percentile and/or reduce the high percentile.
if ~exist('velMagPercentiles','var') || isempty(velMagPercentiles)
    velMagPercentiles = [5 95];  % [low high] in percent
end

% Optional: convert velocity units to mm/s (leave empty to keep in "grid units/frame")
frameRateHz = [];         % frames per second
lambda_mm  = [];          % mm per wavelength (ULM convention), if applicable

%% ===================== SETUP =====================
if ~exist(ulmToolboxDir,'dir')
    error('ulmToolboxDir not found: %s', ulmToolboxDir);
end
if ~exist(trackMatFile,'file')
    error('trackMatFile not found: %s', trackMatFile);
end
if ~exist(outputDir,'dir')
    mkdir(outputDir);
end

addpath(ulmToolboxDir);

%% ===================== LOAD TRACKS =====================
T = load(trackMatFile, 'Track_tot_1', 'Track_tot_2', 'Track_tot_3');

% Collect whichever Track_tot_* variables exist (avoid extra copies / concatenations).
trackFields = {'Track_tot_1','Track_tot_2','Track_tot_3'};
TrackGroups = {};
for i = 1:numel(trackFields)
    fn = trackFields{i};
    if isfield(T, fn) && ~isempty(T.(fn))
        TrackGroups{end+1} = T.(fn); %#ok<AGROW>
    end
end

if isempty(TrackGroups)
    error('No Track_tot_1/2/3 variables found (or they are empty) in %s', trackMatFile);
end

% Count tracks (cell elements) without copying them.
nTracks = 0;
for gi = 1:numel(TrackGroups)
    nTracks = nTracks + numel(TrackGroups{gi});
end
if nTracks == 0
    error('No tracks found in %s', trackMatFile);
end

%% ===================== BUILD RENDER GRID =====================
% Compute bounding box in one pass (avoid slow growing concatenations).
x0 = inf; z0 = inf;
x1 = -inf; z1 = -inf;
for gi = 1:numel(TrackGroups)
    G = TrackGroups{gi};
    for k = 1:numel(G)
        tr = G{k}; % linear indexing works for any cell shape
        if isempty(tr), continue; end
        x0 = min(x0, min(tr(:,1)));
        x1 = max(x1, max(tr(:,1)));
        z0 = min(z0, min(tr(:,2)));
        z1 = max(z1, max(tr(:,2)));
    end
end
if ~isfinite(x0) || ~isfinite(z0)
    error('All tracks are empty in %s', trackMatFile);
end

SRscale = 1 / res;  % track-units per SR pixel

sr_w = ceil((x1 - x0) / SRscale) + 1 + 2*padding_pixels;
sr_h = ceil((z1 - z0) / SRscale) + 1 + 2*padding_pixels;

ULM = struct();
ULM.res = res;
ULM.SRscale = SRscale;
ULM.SRsize = [sr_h, sr_w];

% Axes (in original track units)
llx = (0:ULM.SRsize(2)) * ULM.SRscale + (x0 - padding_pixels*SRscale);
llz = (0:ULM.SRsize(1)) * ULM.SRscale + (z0 - padding_pixels*SRscale);

%% ===================== CONVERT TRACKS FOR ULM_Track2MatOut =====================
% Export format: [x, z, vx, vz, timeline]
% ULM_Track2MatOut expects: [z_idx, x_idx, vz_grid, vx_grid]
Track_matout = cell(nTracks, 1);
outIdx = 1;
for gi = 1:numel(TrackGroups)
    G = TrackGroups{gi};
    for k = 1:numel(G)
        tr = G{k};
        if isempty(tr)
            Track_matout{outIdx} = zeros(0,4);
            outIdx = outIdx + 1;
            continue;
        end
        x  = tr(:,1);
        z  = tr(:,2);
        vx = tr(:,3);
        vz = tr(:,4);

        x_idx = (x - (x0 - padding_pixels*SRscale)) / SRscale + 1;  % 1-indexed
        z_idx = (z - (z0 - padding_pixels*SRscale)) / SRscale + 1;  % 1-indexed

        % Convert velocities to SR pixels per frame for consistent rendering
        vx_grid = vx / SRscale;
        vz_grid = vz / SRscale;

        Track_matout{outIdx} = [z_idx, x_idx, vz_grid, vx_grid];
        outIdx = outIdx + 1;
    end
end

% Free the loaded MAT struct early (can significantly reduce peak memory).
clear T TrackGroups

%% ===================== BUILD MatOut VOLUMES =====================
sizeOut = ULM.SRsize + [1 1]*1;
MatOut      = ULM_Track2MatOut(Track_matout, sizeOut);
MatOut_zdir = ULM_Track2MatOut(Track_matout, sizeOut, 'mode', '2D_vel_z');
MatOut_vel  = ULM_Track2MatOut(Track_matout, sizeOut, 'mode', '2D_velnorm');

velUnitLabel = 'grid units/frame';
if ~isempty(frameRateHz) && ~isempty(lambda_mm)
    MatOut_vel = MatOut_vel * lambda_mm * frameRateHz; % => mm/s
    velUnitLabel = 'mm/s';
end

%% ===================== SAVE .MAT OUTPUTS =====================
[~, trackBase, ~] = fileparts(trackMatFile);
save(fullfile(outputDir, [trackBase '_matouts.mat']), 'MatOut', 'MatOut_zdir', 'MatOut_vel', 'ULM', 'llx', 'llz');

%% ===================== VISUALIZATION =====================
% Note: requires Image Processing Toolbox for imgaussfilt (optional).

% (a) Intensity
figure(1); clf; set(gcf,'Color','w');
IntPower = 1/3;
if swapXZ
    imagesc(llz, llx, (MatOut.^IntPower).'); axis image
else
    imagesc(llx, llz, MatOut.^IntPower); axis image
end
title('ULM intensity');
colormap(gca, gray(128));
colorbar;
if swapXZ
    xlabel('z (track units)'); ylabel('x (track units)');
else
    xlabel('x (track units)'); ylabel('z (track units)');
end
print(gcf, fullfile(outputDir, [trackBase '_MatOut_intensity.png']), '-dpng', '-r300');

% (b) Direction (sign of z velocity)
figure(2); clf; set(gcf,'Color','w');
velCmap = cat(1, flip(flip(hot(128),1),2), hot(128)); velCmap = velCmap(5:end-5,:);
zdir_sm = MatOut_zdir;
try
    zdir_sm = imgaussfilt(MatOut_zdir, .8);
catch
    % imgaussfilt unavailable: use raw
end
ZdirC = (MatOut).^0.25 .* sign(zdir_sm);
if swapXZ
    ZdirC = ZdirC.';
    mask = (MatOut > 0).';
    im = imagesc(llz, llx, ZdirC);
else
    mask = (MatOut > 0);
    im = imagesc(llx, llz, ZdirC);
end
% Ensure background is truly black: mask out zero-intensity pixels.
ax2 = gca;
ax2.Color = 'k';
im.AlphaData = double(mask);
im.CData = im.CData - sign(im.CData)/2; axis image
title('ULM intensity + axial flow direction (sign)');
colormap(gca, velCmap);
colorbar;
if swapXZ
    xlabel('z (track units)'); ylabel('x (track units)');
else
    xlabel('x (track units)'); ylabel('z (track units)');
end
print(gcf, fullfile(outputDir, [trackBase '_MatOut_zdir.png']), '-dpng', '-r300');

% (c) Velocity magnitude map
figure(3); clf; set(gcf,'Color','w');
nonzero = MatOut_vel(MatOut_vel>0);
if isempty(nonzero) || ~any(isfinite(nonzero))
    vLo = 0;
    vHi = 1;
else
    pLo = max(0, min(100, velMagPercentiles(1))) / 100;
    pHi = max(0, min(100, velMagPercentiles(end))) / 100;
    if pHi <= pLo, pLo = 0.05; pHi = 0.95; end
    vLo = max(0, quantile(nonzero, pLo));
    vHi = quantile(nonzero, pHi);
    if ~isfinite(vLo), vLo = 0; end
    if ~isfinite(vHi) || vHi <= vLo
        vLo = 0;
        vHi = max(nonzero);
        if ~isfinite(vHi) || vHi <= 0, vHi = 1; end
    end
end
cmapVel = jet(256);
Mnorm = (MatOut_vel - vLo) / max((vHi - vLo), eps);
Mnorm(Mnorm<0) = 0; Mnorm(Mnorm>1) = 1;
try
    Mnorm = imgaussfilt(Mnorm,.5);
catch
end
Mnorm = Mnorm.^(1/1.5);
idx = round(Mnorm * (size(cmapVel,1)-1)) + 1;
Mrgb = ind2rgb(idx, cmapVel);
Shadow = MatOut; Shadow = Shadow./max(Shadow(:)*.3); Shadow(Shadow>1)=1;
Mrgb = Mrgb .* (Shadow.^(1/4));
Mrgb = brighten(Mrgb,.4);
if swapXZ
    Mrgb = permute(Mrgb, [2 1 3]);
    axX = llz; axY = llx;
else
    axX = llx; axY = llz;
end
axImg = gca;
imshow(Mrgb,'XData',axX,'YData',axY); axis on; axis image
title(sprintf('Velocity magnitude (%g–%g %s)', vLo, vHi, velUnitLabel));

% Put colorbar to the side (like figures 1/2) and make it match the RGB mapping.
colormap(axImg, cmapVel);
caxis(axImg, [vLo vHi]);
cb = colorbar(axImg, 'Location', 'eastoutside');
cb.Label.String = sprintf('Velocity magnitude (%s)', velUnitLabel);

if swapXZ
    xlabel(axImg, 'z (track units)'); ylabel(axImg, 'x (track units)');
else
    xlabel(axImg, 'x (track units)'); ylabel(axImg, 'z (track units)');
end
print(gcf, fullfile(outputDir, [trackBase '_VelMag.png']), '-dpng', '-r300');



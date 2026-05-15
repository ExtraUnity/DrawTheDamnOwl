@echo off
setlocal

rem Autoencoder transition experiment sweep runner
rem Run this from the repo root:
rem   scripts\run_autoencoder_transition_experiments.cmd
rem
rem Example latent U-Net + stage conditioning run:
rem   python experiments\train_autoencoder_transition.py --data_dir data\owl_output --pairs_csv data\owl_output\learning\manifest_transitions.csv --image_size 256 --channels 1 --batch_size 16 --epochs_autoencoder 100 --epochs_transition 100 --lr 1e-3 --latent_channels 64 --latent_noise_std 0.01 --lambda_image 2.0 --transition_model unet --transition_base_channels 64 --transition_bottleneck_blocks 2 --use_stage_conditioning --stage_embed_dim 16 --device cuda --output_dir data\owl_output\learning\autoencoder_transition_unet_stagecond

set "PYTHON_CMD=python"
set "DATA_DIR=data\owl_output"
set "PAIRS_CSV=data\owl_output\learning\manifest_transitions.csv"
set "OUTPUT_ROOT=data\owl_output\learning"
set "IMAGE_SIZE=256"
set "CHANNELS=1"
set "BATCH_SIZE=16"
set "EPOCHS_AUTOENCODER=100"
set "EPOCHS_TRANSITION=100"
set "LR=1e-3"
set "LATENT_NOISE_STD=0.05"
set "FOREGROUND_THRESHOLD=0.05"
set "TRANSITION_ONLY=0"
set "AUTOENCODER_CHECKPOINT="

set "COMMON_ARGS=--data_dir %DATA_DIR% --pairs_csv %PAIRS_CSV% --image_size %IMAGE_SIZE% --channels %CHANNELS% --batch_size %BATCH_SIZE% --epochs_autoencoder %EPOCHS_AUTOENCODER% --epochs_transition %EPOCHS_TRANSITION% --lr %LR% --latent_noise_std %LATENT_NOISE_STD% --foreground_threshold %FOREGROUND_THRESHOLD%"
set "OPTIONAL_ARGS="

if "%TRANSITION_ONLY%"=="1" (
    if "%AUTOENCODER_CHECKPOINT%"=="" (
        echo.
        echo TRANSITION_ONLY is enabled but AUTOENCODER_CHECKPOINT is not set.
        exit /b 1
    )
    set "OPTIONAL_ARGS=--transition_only --autoencoder_checkpoint ""%AUTOENCODER_CHECKPOINT%"""
)

echo.
echo ============================================================
echo Running baseline
echo ============================================================
%PYTHON_CMD% experiments\train_autoencoder_transition.py %COMMON_ARGS% %OPTIONAL_ARGS% --latent_channels 64 --lambda_image 1.0 --output_dir %OUTPUT_ROOT%\autoencoder_transition_baseline
if errorlevel 1 goto :baseline_failed

echo.
echo ============================================================
echo Running c128
echo ============================================================
%PYTHON_CMD% experiments\train_autoencoder_transition.py %COMMON_ARGS% %OPTIONAL_ARGS% --latent_channels 128 --lambda_image 1.0 --output_dir %OUTPUT_ROOT%\autoencoder_transition_c128
if errorlevel 1 goto :c128_failed

echo.
echo ============================================================
echo Running latent U-Net
echo ============================================================
%PYTHON_CMD% experiments\train_autoencoder_transition.py %COMMON_ARGS% %OPTIONAL_ARGS% --latent_channels 64 --lambda_image 2.0 --transition_model unet --transition_base_channels 64 --transition_bottleneck_blocks 2 --output_dir %OUTPUT_ROOT%\autoencoder_transition_unet
if errorlevel 1 goto :unet_failed

echo.
echo ============================================================
echo Running latent U-Net + stage conditioning
echo ============================================================
%PYTHON_CMD% experiments\train_autoencoder_transition.py %COMMON_ARGS% %OPTIONAL_ARGS% --latent_channels 64 --lambda_image 2.0 --transition_model unet --transition_base_channels 64 --transition_bottleneck_blocks 2 --use_stage_conditioning --stage_embed_dim 16 --output_dir %OUTPUT_ROOT%\autoencoder_transition_unet_stagecond
if errorlevel 1 goto :unet_stagecond_failed

echo.
echo All autoencoder transition experiments completed successfully.
exit /b 0

:baseline_failed
echo.
echo Experiment baseline failed.
exit /b 1

:c128_failed
echo.
echo Experiment c128 failed.
exit /b 1

:unet_failed
echo.
echo Experiment latent U-Net failed.
exit /b 1

:unet_stagecond_failed
echo.
echo Experiment latent U-Net + stage conditioning failed.
exit /b 1

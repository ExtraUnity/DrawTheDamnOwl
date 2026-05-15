@echo off
setlocal

rem Autoencoder transition experiment sweep runner
rem Run this from the repo root:
rem   scripts\run_autoencoder_transition_experiments.cmd

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

set "COMMON_ARGS=--data_dir %DATA_DIR% --pairs_csv %PAIRS_CSV% --image_size %IMAGE_SIZE% --channels %CHANNELS% --batch_size %BATCH_SIZE% --epochs_autoencoder %EPOCHS_AUTOENCODER% --epochs_transition %EPOCHS_TRANSITION% --lr %LR% --latent_noise_std %LATENT_NOISE_STD% --foreground_threshold %FOREGROUND_THRESHOLD%"

@REM echo.
@REM echo ============================================================
@REM echo Running baseline
@REM echo ============================================================
@REM %PYTHON_CMD% experiments\train_autoencoder_transition.py %COMMON_ARGS% --latent_channels 64 --lambda_image 1.0 --output_dir %OUTPUT_ROOT%\autoencoder_transition_baseline
@REM if errorlevel 1 goto :baseline_failed

echo.
echo ============================================================
echo Running c128
echo ============================================================
%PYTHON_CMD% experiments\train_autoencoder_transition.py %COMMON_ARGS% --latent_channels 128 --lambda_image 1.0 --output_dir %OUTPUT_ROOT%\autoencoder_transition_c128
if errorlevel 1 goto :residual_fg_edge_c128_failed

@REM echo.
@REM echo ============================================================
@REM echo Running residual_fg_edge_finetune
@REM echo ============================================================
@REM %PYTHON_CMD% experiments\train_autoencoder_transition.py %COMMON_ARGS% --latent_channels 64 --lambda_image 2.0 --lambda_foreground 3.0 --lambda_edge 0.5 --transition_residual --finetune_decoder_epochs 15 --finetune_transition_lr 1e-4 --finetune_decoder_lr 1e-5 --output_dir %OUTPUT_ROOT%\autoencoder_transition_residual_finetune
@REM if errorlevel 1 goto :residual_fg_edge_finetune_failed

echo.
echo All autoencoder transition experiments completed successfully.
exit /b 0

:baseline_failed
echo.
echo Experiment baseline failed.
exit /b 1

:foreground_edge_failed
echo.
echo Experiment foreground_edge failed.
exit /b 1

:residual_fg_edge_failed
echo.
echo Experiment residual_fg_edge failed.
exit /b 1

:residual_fg_edge_c128_failed
echo.
echo Experiment residual_fg_edge_c128 failed.
exit /b 1

:residual_fg_edge_finetune_failed
echo.
echo Experiment residual_fg_edge_finetune failed.
exit /b 1

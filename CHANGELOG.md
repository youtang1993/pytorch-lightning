# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).


## [unreleased] - YYYY-MM-DD

### Added

- Enable PyTorch 1.7 compatibility ([#3541](https://github.com/PyTorchLightning/pytorch-lightning/pull/3541))

- Added hooks to metric module interface ([#2528](https://github.com/PyTorchLightning/pytorch-lightning/pull/2528))

- Added `LightningModule.to_torchscript` to support exporting as `ScriptModule` ([#3258](https://github.com/PyTorchLightning/pytorch-lightning/pull/3258/))

- Added `EmbeddingSimilarity` metric:
   * functional interface ([#3349](https://github.com/PyTorchLightning/pytorch-lightning/pull/3349))
   * class based interface + tests ([#3358](https://github.com/PyTorchLightning/pytorch-lightning/pull/3358))

- Added warning when dropping unpicklable `hparams` ([#2874](https://github.com/PyTorchLightning/pytorch-lightning/pull/2874))

- Added error when AUROC metric is used for multiclass problems ([#3350](https://github.com/PyTorchLightning/pytorch-lightning/pull/3350))

- Allow `ModelCheckpoint` monitor to be `None`, meaning it will always save ([3630](https://github.com/PyTorchLightning/pytorch-lightning/pull/3630))

- Disabled optimizers setup during testing ([#3059](https://github.com/PyTorchLightning/pytorch-lightning/pull/3059))

- Added support for datamodules to save and load checkpoints when training ([#3563]https://github.com/PyTorchLightning/pytorch-lightning/pull/3563)

- Added support for datamodule in learning rate finder ([#3425](https://github.com/PyTorchLightning/pytorch-lightning/pull/3425))

- Added `broadcast` to `TPUBackend` ([#3814](https://github.com/PyTorchLightning/pytorch-lightning/pull/3814))

### Changed

- Refactored accelerator backends:
   * moved TPU `xxx_step` to backend ([#3118](https://github.com/PyTorchLightning/pytorch-lightning/pull/3118))
   * refactored DDP backend `forward` ([#3119](https://github.com/PyTorchLightning/pytorch-lightning/pull/3119))
   * refactored GPU backend `__step` ([#3120](https://github.com/PyTorchLightning/pytorch-lightning/pull/3120))
   * refactored Horovod backend ([#3121](https://github.com/PyTorchLightning/pytorch-lightning/pull/3121),
        [#3122](https://github.com/PyTorchLightning/pytorch-lightning/pull/3122))
   * remove obscure forward call in eval + CPU backend `___step` ([#3123](https://github.com/PyTorchLightning/pytorch-lightning/pull/3123))
   * reduced all simplified forward ([#3126](https://github.com/PyTorchLightning/pytorch-lightning/pull/3126))
   * added hook base method ([#3127](https://github.com/PyTorchLightning/pytorch-lightning/pull/3127))
   * refactor eval loop to use hooks - use `test_mode` for if so we can split later ([#3129](https://github.com/PyTorchLightning/pytorch-lightning/pull/3129))
   * moved `___step_end` hooks ([#3130](https://github.com/PyTorchLightning/pytorch-lightning/pull/3130))
   * training forward refactor ([#3134](https://github.com/PyTorchLightning/pytorch-lightning/pull/3134))
   * training AMP scaling refactor ([#3135](https://github.com/PyTorchLightning/pytorch-lightning/pull/3135))
   * eval step scaling factor ([#3136](https://github.com/PyTorchLightning/pytorch-lightning/pull/3136))
   * add eval loop object to streamline eval loop ([#3138](https://github.com/PyTorchLightning/pytorch-lightning/pull/3138))
   * refactored dataloader process hook ([#3139](https://github.com/PyTorchLightning/pytorch-lightning/pull/3139))
   * refactored inner eval loop ([#3141](https://github.com/PyTorchLightning/pytorch-lightning/pull/3141))
   * final inner eval loop hooks ([#3154](https://github.com/PyTorchLightning/pytorch-lightning/pull/3154))
   * clean up hooks in `run_evaluation` ([#3156](https://github.com/PyTorchLightning/pytorch-lightning/pull/3156))
   * clean up data reset ([#3161](https://github.com/PyTorchLightning/pytorch-lightning/pull/3161))
   * expand eval loop out ([#3165](https://github.com/PyTorchLightning/pytorch-lightning/pull/3165))
   * moved hooks around in eval loop ([#3195](https://github.com/PyTorchLightning/pytorch-lightning/pull/3195))
   * remove `_evaluate` fx ([#3197](https://github.com/PyTorchLightning/pytorch-lightning/pull/3197))
   * `Trainer.fit` hook clean up ([#3198](https://github.com/PyTorchLightning/pytorch-lightning/pull/3198))
   * DDPs train hooks ([#3203](https://github.com/PyTorchLightning/pytorch-lightning/pull/3203))
   * refactor DDP backend ([#3204](https://github.com/PyTorchLightning/pytorch-lightning/pull/3204),
        [#3207](https://github.com/PyTorchLightning/pytorch-lightning/pull/3207),
        [#3208](https://github.com/PyTorchLightning/pytorch-lightning/pull/3208),
        [#3209](https://github.com/PyTorchLightning/pytorch-lightning/pull/3209),
        [#3210](https://github.com/PyTorchLightning/pytorch-lightning/pull/3210))
   * reduced accelerator selection ([#3211](https://github.com/PyTorchLightning/pytorch-lightning/pull/3211))
   * group prepare data hook ([#3212](https://github.com/PyTorchLightning/pytorch-lightning/pull/3212))

- Changed `LearningRateLogger` to `LearningRateMonitor` ([#3251](https://github.com/PyTorchLightning/pytorch-lightning/pull/3251))

- Used `fsspec` instead of `gfile` for all IO ([#3320](https://github.com/PyTorchLightning/pytorch-lightning/pull/3320))

- Refactor `GPUStatsMonitor` to improve training speed ([#3257](https://github.com/PyTorchLightning/pytorch-lightning/pull/3257))

- Renamed `reduction` to `class_reduction` in classification metrics ([#3322](https://github.com/PyTorchLightning/pytorch-lightning/pull/3322))

- Changed `class_reduction` similar to sklearn for classification metrics ([#3322](https://github.com/PyTorchLightning/pytorch-lightning/pull/3322))

- Changed IoU score behavior for classes absent in target and pred ([#3098](https://github.com/PyTorchLightning/pytorch-lightning/pull/3098))

- Changed IoU `remove_bg` bool to `ignore_index` optional int ([#3098](https://github.com/PyTorchLightning/pytorch-lightning/pull/3098))

- Changed defaults of `save_top_k` and `save_last` to `None` in ModelCheckpoint ([#3680](https://github.com/PyTorchLightning/pytorch-lightning/pull/3680))

- `row_log_interval` and `log_save_interval` are now based on training loop's `global_step` instead of epoch-internal batch index ([#3667](https://github.com/PyTorchLightning/pytorch-lightning/pull/3667))

- Swap `torch.load` for `fsspec` load in DDP spawn backend ([#3787](https://github.com/PyTorchLightning/pytorch-lightning/pull/3787))

- Swap `torch.load` for `fsspec` load in cloud_io loading ([#3692](https://github.com/PyTorchLightning/pytorch-lightning/pull/3692))

### Deprecated


### Removed


### Fixed

- Fixed `on_train_batch_start` hook to end epoch early ([#3700](https://github.com/PyTorchLightning/pytorch-lightning/pull/3700))

- Fixed `num_sanity_val_steps` is clipped to `limit_val_batches` ([#2917](https://github.com/PyTorchLightning/pytorch-lightning/pull/2917))

- Fixed RMSLE metric ([#3188](https://github.com/PyTorchLightning/pytorch-lightning/pull/3188))

- Fixed ONNX model save on GPU ([#3145](https://github.com/PyTorchLightning/pytorch-lightning/pull/3145))

- Fixed `GpuUsageLogger` to work on different platforms ([#3008](https://github.com/PyTorchLightning/pytorch-lightning/pull/3008))

- Fixed auto-scale batch size not dumping `auto_lr_find` parameter ([#3151](https://github.com/PyTorchLightning/pytorch-lightning/pull/3151))

- Fixed `batch_outputs` with optimizer frequencies ([#3229](https://github.com/PyTorchLightning/pytorch-lightning/pull/3229))

- Fixed setting batch size in `LightningModule.datamodule` when using `auto_scale_batch_size` ([#3266](https://github.com/PyTorchLightning/pytorch-lightning/pull/3266))

- Fixed Horovod distributed backend compatibility with native AMP ([#3404](https://github.com/PyTorchLightning/pytorch-lightning/pull/3404))

- Fixed batch size auto scaling exceeding the size of the dataset ([#3271](https://github.com/PyTorchLightning/pytorch-lightning/pull/3271))

- Fixed getting `experiment_id` from MLFlow only once instead of each training loop ([#3394](https://github.com/PyTorchLightning/pytorch-lightning/pull/3394))

- Fixed overfit_batches which now correctly disables shuffling for the training loader. ([#3501](https://github.com/PyTorchLightning/pytorch-lightning/pull/3501))

- Fixed gradient norm tracking for `row_log_interval > 1` ([#3489](https://github.com/PyTorchLightning/pytorch-lightning/pull/3489))

- Fixed `ModelCheckpoint` name formatting ([3164](https://github.com/PyTorchLightning/pytorch-lightning/pull/3163))

- Fixed auto-scale batch size ([#3151](https://github.com/PyTorchLightning/pytorch-lightning/pull/3151))

- Fixed example implementation of AutoEncoder ([#3190](https://github.com/PyTorchLightning/pytorch-lightning/pull/3190))

- Fixed invalid paths when remote logging with TensorBoard ([#3236](https://github.com/PyTorchLightning/pytorch-lightning/pull/3236))

- Fixed dataloader shuffling not getting turned off with `overfit_batches > 0` and `distributed_backend = "ddp"` ([#3534](https://github.com/PyTorchLightning/pytorch-lightning/pull/3534))

- Fixed determinism in `DDPSpawnBackend` when using `seed_everything` in main process ([#3335](https://github.com/PyTorchLightning/pytorch-lightning/pull/3335))

- Fixed `ModelCheckpoint` `period` to actually save every `period` epochs ([#3630](https://github.com/PyTorchLightning/pytorch-lightning/pull/3630))

- Fixed `val_progress_bar` total with `num_sanity_val_steps` ([#3751](https://github.com/PyTorchLightning/pytorch-lightning/pull/3751))

- Fixed `ModelCheckpoint` with `save_top_k=-1` option not tracking the best models when a monitor metric is available ([#3735](https://github.com/PyTorchLightning/pytorch-lightning/pull/3735))

- Fixed counter-intuitive error being thrown in `Accuracy` metric for zero target tensor ([#3764](https://github.com/PyTorchLightning/pytorch-lightning/pull/3764))

- Fixed aggregation of metrics ([#3517](https://github.com/PyTorchLightning/pytorch-lightning/pull/3517))

- Fixed `current_epoch` and `global_step` properties mismatch between `Trainer` and `LightningModule` ([#3785](https://github.com/PyTorchLightning/pytorch-lightning/pull/3785))

## [0.9.0] - 2020-08-20

### Added

- Added SyncBN for DDP ([#2801](https://github.com/PyTorchLightning/pytorch-lightning/pull/2801),
     [#2838](https://github.com/PyTorchLightning/pytorch-lightning/pull/2838))
- Added basic `CSVLogger` ([#2721](https://github.com/PyTorchLightning/pytorch-lightning/pull/2721))
- Added SSIM metrics ([#2671](https://github.com/PyTorchLightning/pytorch-lightning/pull/2671))
- Added BLEU metrics ([#2535](https://github.com/PyTorchLightning/pytorch-lightning/pull/2535))
- Added support to export a model to ONNX format ([#2596](https://github.com/PyTorchLightning/pytorch-lightning/pull/2596))
- Added support for `Trainer(num_sanity_val_steps=-1)` to check all validation data before training ([#2246](https://github.com/PyTorchLightning/pytorch-lightning/pull/2246))
- Added struct. output:
  * tests for val loop flow ([#2605](https://github.com/PyTorchLightning/pytorch-lightning/pull/2605))
  * `EvalResult` support for train and val. loop ([#2615](https://github.com/PyTorchLightning/pytorch-lightning/pull/2615),
       [#2651](https://github.com/PyTorchLightning/pytorch-lightning/pull/2651))
  * weighted average in results obj ([#2930](https://github.com/PyTorchLightning/pytorch-lightning/pull/2930))
  * fix result obj DP auto reduce ([#3013](https://github.com/PyTorchLightning/pytorch-lightning/pull/3013))
- Added class `LightningDataModule` ([#2668](https://github.com/PyTorchLightning/pytorch-lightning/pull/2668))
- Added support for PyTorch 1.6 ([#2745](https://github.com/PyTorchLightning/pytorch-lightning/pull/2745))
- Added call DataModule hooks implicitly in trainer ([#2755](https://github.com/PyTorchLightning/pytorch-lightning/pull/2755))
- Added support for Mean in DDP Sync ([#2568](https://github.com/PyTorchLightning/pytorch-lightning/pull/2568))
- Added remaining `sklearn` metrics: `AveragePrecision`, `BalancedAccuracy`, `CohenKappaScore`, `DCG`, `Hamming`, `Hinge`, `Jaccard`, `MeanAbsoluteError`, `MeanSquaredError`, `MeanSquaredLogError`, `MedianAbsoluteError`, `R2Score`, `MeanPoissonDeviance`, `MeanGammaDeviance`, `MeanTweedieDeviance`, `ExplainedVariance` ([#2562](https://github.com/PyTorchLightning/pytorch-lightning/pull/2562))
- Added support for `limit_{mode}_batches (int)` to work with infinite dataloader (IterableDataset) ([#2840](https://github.com/PyTorchLightning/pytorch-lightning/pull/2840))
- Added support returning python scalars in DP ([#1935](https://github.com/PyTorchLightning/pytorch-lightning/pull/1935))
- Added support to Tensorboard logger for OmegaConf `hparams` ([#2846](https://github.com/PyTorchLightning/pytorch-lightning/pull/2846))
- Added tracking of basic states in `Trainer` ([#2541](https://github.com/PyTorchLightning/pytorch-lightning/pull/2541))
- Tracks all outputs including TBPTT and multiple optimizers ([#2890](https://github.com/PyTorchLightning/pytorch-lightning/pull/2890))
- Added GPU Usage Logger ([#2932](https://github.com/PyTorchLightning/pytorch-lightning/pull/2932))
- Added `strict=False` for `load_from_checkpoint` ([#2819](https://github.com/PyTorchLightning/pytorch-lightning/pull/2819))
- Added saving test predictions on multiple GPUs ([#2926](https://github.com/PyTorchLightning/pytorch-lightning/pull/2926))
- Auto log the computational graph for loggers that support this ([#3003](https://github.com/PyTorchLightning/pytorch-lightning/pull/3003))
- Added warning when changing monitor and using results obj ([#3014](https://github.com/PyTorchLightning/pytorch-lightning/pull/3014))
- Added a hook `transfer_batch_to_device` to the `LightningDataModule` ([#3038](https://github.com/PyTorchLightning/pytorch-lightning/pull/3038))

### Changed

- Truncated long version numbers in progress bar ([#2594](https://github.com/PyTorchLightning/pytorch-lightning/pull/2594))
- Enabling val/test loop disabling ([#2692](https://github.com/PyTorchLightning/pytorch-lightning/pull/2692))
- Refactored into `accelerator` module:
    * GPU training ([#2704](https://github.com/PyTorchLightning/pytorch-lightning/pull/2704))
    * TPU training ([#2708](https://github.com/PyTorchLightning/pytorch-lightning/pull/2708))
    * DDP(2) backend ([#2796](https://github.com/PyTorchLightning/pytorch-lightning/pull/2796))
    * Retrieve last logged val from result by key ([#3049](https://github.com/PyTorchLightning/pytorch-lightning/pull/3049))
- Using `.comet.config` file for `CometLogger` ([#1913](https://github.com/PyTorchLightning/pytorch-lightning/pull/1913))
- Updated hooks arguments - breaking for `setup` and `teardown` ([#2850](https://github.com/PyTorchLightning/pytorch-lightning/pull/2850))
- Using `gfile` to support remote directories ([#2164](https://github.com/PyTorchLightning/pytorch-lightning/pull/2164))
- Moved optimizer creation after device placement for DDP backends ([#2904](https://github.com/PyTorchLightning/pytorch-lighting/pull/2904))
- Support `**DictConfig` for `hparam` serialization ([#2519](https://github.com/PyTorchLightning/pytorch-lightning/pull/2519))
- Removed callback metrics from test results obj ([#2994](https://github.com/PyTorchLightning/pytorch-lightning/pull/2994))
- Re-enabled naming metrics in ckpt name ([#3060](https://github.com/PyTorchLightning/pytorch-lightning/pull/3060))
- Changed progress bar epoch counting to start from 0 ([#3061](https://github.com/PyTorchLightning/pytorch-lightning/pull/3061))

### Deprecated

- Deprecated Trainer attribute `ckpt_path`, which will now be set by `weights_save_path` ([#2681](https://github.com/PyTorchLightning/pytorch-lightning/pull/2681))

### Removed

- Removed deprecated: ([#2760](https://github.com/PyTorchLightning/pytorch-lightning/pull/2760))
    * core decorator `data_loader`
    * Module hook `on_sanity_check_start` and loading `load_from_metrics`
    * package `pytorch_lightning.logging`
    * Trainer arguments: `show_progress_bar`, `num_tpu_cores`, `use_amp`, `print_nan_grads`
    * LR Finder argument `num_accumulation_steps`

### Fixed

- Fixed `accumulate_grad_batches` for last batch ([#2853](https://github.com/PyTorchLightning/pytorch-lightning/pull/2853))
- Fixed setup call while testing ([#2624](https://github.com/PyTorchLightning/pytorch-lightning/pull/2624))
- Fixed local rank zero casting ([#2640](https://github.com/PyTorchLightning/pytorch-lightning/pull/2640))
- Fixed single scalar return from training ([#2587](https://github.com/PyTorchLightning/pytorch-lightning/pull/2587))
- Fixed Horovod backend to scale LR schedlers with the optimizer ([#2626](https://github.com/PyTorchLightning/pytorch-lightning/pull/2626))
- Fixed `dtype` and `device` properties not getting updated in submodules ([#2657](https://github.com/PyTorchLightning/pytorch-lightning/pull/2657))
- Fixed `fast_dev_run` to run for all dataloaders ([#2581](https://github.com/PyTorchLightning/pytorch-lightning/pull/2581))
- Fixed `save_dir` in loggers getting ignored by default value of `weights_save_path` when user did not specify `weights_save_path` ([#2681](https://github.com/PyTorchLightning/pytorch-lightning/pull/2681))
- Fixed `weights_save_path` getting ignored when `logger=False` is passed to Trainer ([#2681](https://github.com/PyTorchLightning/pytorch-lightning/pull/2681))
- Fixed TPU multi-core and Float16 ([#2632](https://github.com/PyTorchLightning/pytorch-lightning/pull/2632))
- Fixed test metrics not being logged with `LoggerCollection` ([#2723](https://github.com/PyTorchLightning/pytorch-lightning/pull/2723))
- Fixed data transfer to device when using `torchtext.data.Field` and `include_lengths is True` ([#2689](https://github.com/PyTorchLightning/pytorch-lightning/pull/2689))
- Fixed shuffle argument for distributed sampler ([#2789](https://github.com/PyTorchLightning/pytorch-lightning/pull/2789))
- Fixed logging interval ([#2694](https://github.com/PyTorchLightning/pytorch-lightning/pull/2694))
- Fixed loss value in the progress bar is wrong when `accumulate_grad_batches > 1` ([#2738](https://github.com/PyTorchLightning/pytorch-lightning/pull/2738))
- Fixed correct CWD for ddp sub-processes when using Hydra ([#2719](https://github.com/PyTorchLightning/pytorch-lightning/pull/2719))
- Fixed selecting GPUs using `CUDA_VISIBLE_DEVICES` ([#2739](https://github.com/PyTorchLightning/pytorch-lightning/pull/2739),
     [#2796](https://github.com/PyTorchLightning/pytorch-lightning/pull/2796))
- Fixed false `num_classes` warning in metrics ([#2781](https://github.com/PyTorchLightning/pytorch-lightning/pull/2781))
- Fixed shell injection vulnerability in subprocess call ([#2786](https://github.com/PyTorchLightning/pytorch-lightning/pull/2786))
- Fixed LR finder and `hparams` compatibility ([#2821](https://github.com/PyTorchLightning/pytorch-lightning/pull/2821))
- Fixed `ModelCheckpoint` not saving the latest information when `save_last=True` ([#2881](https://github.com/PyTorchLightning/pytorch-lightning/pull/2881))
- Fixed ImageNet example: learning rate scheduler, number of workers and batch size when using DDP ([#2889](https://github.com/PyTorchLightning/pytorch-lightning/pull/2889))
- Fixed apex gradient clipping ([#2829](https://github.com/PyTorchLightning/pytorch-lightning/pull/2829))
- Fixed save apex scaler states ([#2828](https://github.com/PyTorchLightning/pytorch-lightning/pull/2828))
- Fixed a model loading issue with inheritance and variable positional arguments ([#2911](https://github.com/PyTorchLightning/pytorch-lightning/pull/2911))
- Fixed passing `non_blocking=True` when transferring a batch object that does not support it ([#2910](https://github.com/PyTorchLightning/pytorch-lightning/pull/2910))
- Fixed checkpointing to remote file paths ([#2925](https://github.com/PyTorchLightning/pytorch-lightning/pull/2925))
- Fixed adding val step argument to metrics ([#2986](https://github.com/PyTorchLightning/pytorch-lightning/pull/2986))
- Fixed an issue that caused `Trainer.test()` to stall in ddp mode ([#2997](https://github.com/PyTorchLightning/pytorch-lightning/pull/2997))
- Fixed gathering of results with tensors of varying shape ([#3020](https://github.com/PyTorchLightning/pytorch-lightning/pull/3020))
- Fixed batch size auto-scaling feature to set the new value on the correct model attribute ([#3043](https://github.com/PyTorchLightning/pytorch-lightning/pull/3043)) 
- Fixed automatic batch scaling not working with half precision ([#3045](https://github.com/PyTorchLightning/pytorch-lightning/pull/3045))
- Fixed setting device to root gpu ([#3042](https://github.com/PyTorchLightning/pytorch-lightning/pull/3042))

## [0.8.5] - 2020-07-09

### Added

- Added a PSNR metric: peak signal-to-noise ratio ([#2483](https://github.com/PyTorchLightning/pytorch-lightning/pull/2483))
- Added functional regression metrics ([#2492](https://github.com/PyTorchLightning/pytorch-lightning/pull/2492))

### Removed

- Removed auto val reduce ([#2462](https://github.com/PyTorchLightning/pytorch-lightning/pull/2462))

### Fixed

- Flattening Wandb Hyperparameters ([#2459](https://github.com/PyTorchLightning/pytorch-lightning/pull/2459))
- Fixed using the same DDP python interpreter and actually running ([#2482](https://github.com/PyTorchLightning/pytorch-lightning/pull/2482))
- Fixed model summary input type conversion for models that have input dtype different from model parameters ([#2510](https://github.com/PyTorchLightning/pytorch-lightning/pull/2510))
- Made `TensorBoardLogger` and `CometLogger` pickleable ([#2518](https://github.com/PyTorchLightning/pytorch-lightning/pull/2518))
- Fixed a problem with `MLflowLogger` creating multiple run folders ([#2502](https://github.com/PyTorchLightning/pytorch-lightning/pull/2502))
- Fixed global_step increment ([#2455](https://github.com/PyTorchLightning/pytorch-lightning/pull/2455))
- Fixed TPU hanging example ([#2488](https://github.com/PyTorchLightning/pytorch-lightning/pull/2488))
- Fixed `argparse` default value bug ([#2526](https://github.com/PyTorchLightning/pytorch-lightning/pull/2526))
- Fixed Dice and IoU to avoid NaN by adding small eps ([#2545](https://github.com/PyTorchLightning/pytorch-lightning/pull/2545))
- Fixed accumulate gradients schedule at epoch 0 (continued) ([#2513](https://github.com/PyTorchLightning/pytorch-lightning/pull/2513))
- Fixed Trainer `.fit()` returning last not best weights in "ddp_spawn" ([#2565](https://github.com/PyTorchLightning/pytorch-lightning/pull/2565))
- Fixed passing (do not pass) TPU weights back on test ([#2566](https://github.com/PyTorchLightning/pytorch-lightning/pull/2566))
- Fixed DDP tests and `.test()` ([#2512](https://github.com/PyTorchLightning/pytorch-lightning/pull/2512),
     [#2570](https://github.com/PyTorchLightning/pytorch-lightning/pull/2570))

## [0.8.4] - 2020-07-01

### Added

- Added reduce ddp results on eval ([#2434](https://github.com/PyTorchLightning/pytorch-lightning/pull/2434))
- Added a warning when an `IterableDataset` has `__len__` defined ([#2437](https://github.com/PyTorchLightning/pytorch-lightning/pull/2437))

### Changed

- Enabled no returns from eval ([#2446](https://github.com/PyTorchLightning/pytorch-lightning/pull/2446))

### Fixed

- Fixes train outputs ([#2428](https://github.com/PyTorchLightning/pytorch-lightning/pull/2428))
- Fixes Conda dependencies ([#2412](https://github.com/PyTorchLightning/pytorch-lightning/pull/2412))
- Fixed Apex scaling with decoupled backward ([#2433](https://github.com/PyTorchLightning/pytorch-lightning/pull/2433))
- Fixed crashing or wrong displaying progressbar because of missing ipywidgets ([#2417](https://github.com/PyTorchLightning/pytorch-lightning/pull/2417))
- Fixed TPU saving dir ([fc26078e](https://github.com/PyTorchLightning/pytorch-lightning/commit/fc26078e395f8a001f4c6dd7b3fe7ca202f914a3), [04e68f02](https://github.com/PyTorchLightning/pytorch-lightning/commit/04e68f022fc03dd5f1555ee86dea997d42a448ad))
- Fixed logging on rank 0 only ([#2425](https://github.com/PyTorchLightning/pytorch-lightning/pull/2425))


## [0.8.3] - 2020-06-29

### Fixed

- Fixed AMP wrong call ([593837e](https://github.com/PyTorchLightning/pytorch-lightning/commit/593837e1da24ff6c942b24ed803fc1496a304609))
- Fixed batch typo ([92d1e75](https://github.com/PyTorchLightning/pytorch-lightning/commit/92d1e75b2638a493d9d21ed5fe00a22093888285))

## [0.8.2] - 2020-06-28

### Added

- Added TorchText support for moving data to GPU ([#2379](https://github.com/PyTorchLightning/pytorch-lightning/pull/2379))

### Changed

- Changed epoch indexing from 0 instead of 1 ([#2289](https://github.com/PyTorchLightning/pytorch-lightning/pull/2289))
- Refactor Model `backward` ([#2276](https://github.com/PyTorchLightning/pytorch-lightning/pull/2276))
- Refactored `training_batch` + tests to verify correctness ([#2327](https://github.com/PyTorchLightning/pytorch-lightning/pull/2327),
     [#2328](https://github.com/PyTorchLightning/pytorch-lightning/pull/2328))
- Refactored training loop ([#2336](https://github.com/PyTorchLightning/pytorch-lightning/pull/2336))
- Made optimization steps for hooks ([#2363](https://github.com/PyTorchLightning/pytorch-lightning/pull/2363))
- Changed default apex level to 'O2' ([#2362](https://github.com/PyTorchLightning/pytorch-lightning/pull/2362))

### Removed

- Moved `TrainsLogger` to Bolts ([#2384](https://github.com/PyTorchLightning/pytorch-lightning/pull/2384))

### Fixed

- Fixed parsing TPU arguments and TPU tests ([#2094](https://github.com/PyTorchLightning/pytorch-lightning/pull/2094))
- Fixed number batches in case of multiple dataloaders and `limit_{*}_batches` ([#1920](https://github.com/PyTorchLightning/pytorch-lightning/pull/1920),
     [#2226](https://github.com/PyTorchLightning/pytorch-lightning/pull/2226))
- Fixed an issue with forward hooks not being removed after model summary ([#2298](https://github.com/PyTorchLightning/pytorch-lightning/pull/2298))
- Fix for `load_from_checkpoint()` not working with absolute path on Windows ([#2294](https://github.com/PyTorchLightning/pytorch-lightning/pull/2294))
- Fixed an issue how _has_len handles `NotImplementedError` e.g. raised by `torchtext.data.Iterator` ([#2293](https://github.com/PyTorchLightning/pytorch-lightning/pull/2293)), ([#2307](https://github.com/PyTorchLightning/pytorch-lightning/pull/2307))
- Fixed `average_precision` metric ([#2319](https://github.com/PyTorchLightning/pytorch-lightning/pull/2319))
- Fixed ROC metric for CUDA tensors ([#2304](https://github.com/PyTorchLightning/pytorch-lightning/pull/2304))
- Fixed `average_precision` metric ([#2319](https://github.com/PyTorchLightning/pytorch-lightning/pull/2319))
- Fixed lost compatibility with custom datatypes implementing `.to` ([#2335](https://github.com/PyTorchLightning/pytorch-lightning/pull/2335))
- Fixed loading model with kwargs ([#2387](https://github.com/PyTorchLightning/pytorch-lightning/pull/2387))
- Fixed sum(0) for `trainer.num_val_batches` ([#2268](https://github.com/PyTorchLightning/pytorch-lightning/pull/2268))
- Fixed checking if the parameters are a `DictConfig` Object ([#2216](https://github.com/PyTorchLightning/pytorch-lightning/pull/2216))
- Fixed SLURM weights saving ([#2341](https://github.com/PyTorchLightning/pytorch-lightning/pull/2341))
- Fixed swaps LR scheduler order ([#2356](https://github.com/PyTorchLightning/pytorch-lightning/pull/2356))
- Fixed adding tensorboard `hparams` logging test ([#2342](https://github.com/PyTorchLightning/pytorch-lightning/pull/2342))
- Fixed use model ref for tear down ([#2360](https://github.com/PyTorchLightning/pytorch-lightning/pull/2360))
- Fixed logger crash on DDP ([#2388](https://github.com/PyTorchLightning/pytorch-lightning/pull/2388))
- Fixed several issues with early stopping and checkpoint callbacks ([#1504](https://github.com/PyTorchLightning/pytorch-lightning/pull/1504),
     [#2391](https://github.com/PyTorchLightning/pytorch-lightning/pull/2391))
- Fixed loading past checkpoints from v0.7.x ([#2405](https://github.com/PyTorchLightning/pytorch-lightning/pull/2405))
- Fixed loading model without arguments ([#2403](https://github.com/PyTorchLightning/pytorch-lightning/pull/2403))
- Fixed Windows compatibility issue ([#2358](https://github.com/PyTorchLightning/pytorch-lightning/pull/2358))

## [0.8.1] - 2020-06-19

### Fixed

- Fixed the `load_from_checkpoint` path detected as URL bug ([#2244](https://github.com/PyTorchLightning/pytorch-lightning/pull/2244))
- Fixed hooks - added barrier ([#2245](https://github.com/PyTorchLightning/pytorch-lightning/pull/2245),
     [#2257](https://github.com/PyTorchLightning/pytorch-lightning/pull/2257),
     [#2260](https://github.com/PyTorchLightning/pytorch-lightning/pull/220))
- Fixed `hparams` - remove frame inspection on `self.hparams` ([#2253](https://github.com/PyTorchLightning/pytorch-lightning/pull/2253))
- Fixed setup and on fit calls ([#2252](https://github.com/PyTorchLightning/pytorch-lightning/pull/2252))
- Fixed GPU template ([#2255](https://github.com/PyTorchLightning/pytorch-lightning/pull/2255))

## [0.8.0] - 2020-06-18

### Added

- Added `overfit_batches`, `limit_{val|test}_batches` flags (overfit now uses training set for all three) ([#2213](https://github.com/PyTorchLightning/pytorch-lightning/pull/2213))
- Added metrics
  * Base classes ([#1326](https://github.com/PyTorchLightning/pytorch-lightning/pull/1326),
       [#1877](https://github.com/PyTorchLightning/pytorch-lightning/pull/1877))
  * Sklearn metrics classes ([#1327](https://github.com/PyTorchLightning/pytorch-lightning/pull/1327))
  * Native torch metrics ([#1488](https://github.com/PyTorchLightning/pytorch-lightning/pull/1488),
       [#2062](https://github.com/PyTorchLightning/pytorch-lightning/pull/2062))
  * docs for all Metrics ([#2184](https://github.com/PyTorchLightning/pytorch-lightning/pull/2184),
       [#2209](https://github.com/PyTorchLightning/pytorch-lightning/pull/2209))
  * Regression metrics ([#2221](https://github.com/PyTorchLightning/pytorch-lightning/pull/2221))
- Added type hints in `Trainer.fit()` and `Trainer.test()` to reflect that also a list of dataloaders can be passed in ([#1723](https://github.com/PyTorchLightning/pytorch-lightning/pull/1723))
- Allow dataloaders without sampler field present ([#1907](https://github.com/PyTorchLightning/pytorch-lightning/pull/1907))
- Added option `save_last` to save the model at the end of every epoch in `ModelCheckpoint` [(#1908)](https://github.com/PyTorchLightning/pytorch-lightning/pull/1908)
- Early stopping checks `on_validation_end` ([#1458](https://github.com/PyTorchLightning/pytorch-lightning/pull/1458))
- Attribute `best_model_path` to `ModelCheckpoint` for storing and later retrieving the path to the best saved model file ([#1799](https://github.com/PyTorchLightning/pytorch-lightning/pull/1799))
- Speed up single-core TPU training by loading data using `ParallelLoader` ([#2033](https://github.com/PyTorchLightning/pytorch-lightning/pull/2033))
- Added a model hook `transfer_batch_to_device` that enables moving custom data structures to the target device ([1756](https://github.com/PyTorchLightning/pytorch-lightning/pull/1756))
- Added [black](https://black.readthedocs.io/en/stable/) formatter for the code with code-checker on pull ([1610](https://github.com/PyTorchLightning/pytorch-lightning/pull/1610))
- Added back the slow spawn ddp implementation as `ddp_spawn` ([#2115](https://github.com/PyTorchLightning/pytorch-lightning/pull/2115))
- Added loading checkpoints from URLs ([#1667](https://github.com/PyTorchLightning/pytorch-lightning/pull/1667))
- Added a callback method `on_keyboard_interrupt` for handling KeyboardInterrupt events during training ([#2134](https://github.com/PyTorchLightning/pytorch-lightning/pull/2134))
- Added a decorator `auto_move_data` that moves data to the correct device when using the LightningModule for inference ([#1905](https://github.com/PyTorchLightning/pytorch-lightning/pull/1905))
- Added `ckpt_path` option to `LightningModule.test(...)` to load particular checkpoint ([#2190](https://github.com/PyTorchLightning/pytorch-lightning/pull/2190))
- Added `setup` and `teardown` hooks for model ([#2229](https://github.com/PyTorchLightning/pytorch-lightning/pull/2229))

### Changed

- Allow user to select individual TPU core to train on ([#1729](https://github.com/PyTorchLightning/pytorch-lightning/pull/1729))
- Removed non-finite values from loss in `LRFinder` ([#1862](https://github.com/PyTorchLightning/pytorch-lightning/pull/1862))
- Allow passing model hyperparameters as complete kwarg list ([#1896](https://github.com/PyTorchLightning/pytorch-lightning/pull/1896))
- Renamed `ModelCheckpoint`'s attributes `best` to `best_model_score` and `kth_best_model` to `kth_best_model_path` ([#1799](https://github.com/PyTorchLightning/pytorch-lightning/pull/1799))
- Re-Enable Logger's `ImportError`s ([#1938](https://github.com/PyTorchLightning/pytorch-lightning/pull/1938))
- Changed the default value of the Trainer argument `weights_summary` from `full` to `top` ([#2029](https://github.com/PyTorchLightning/pytorch-lightning/pull/2029))
- Raise an error when lightning replaces an existing sampler ([#2020](https://github.com/PyTorchLightning/pytorch-lightning/pull/2020))
- Enabled `prepare_data` from correct processes - clarify local vs global rank ([#2166](https://github.com/PyTorchLightning/pytorch-lightning/pull/2166))
- Remove explicit flush from tensorboard logger ([#2126](https://github.com/PyTorchLightning/pytorch-lightning/pull/2126))
- Changed epoch indexing from 1 instead of 0 ([#2206](https://github.com/PyTorchLightning/pytorch-lightning/pull/2206))

### Deprecated

- Deprecated flags: ([#2213](https://github.com/PyTorchLightning/pytorch-lightning/pull/2213))
  * `overfit_pct` in favour of `overfit_batches`
  * `val_percent_check` in favour of `limit_val_batches`
  * `test_percent_check` in favour of `limit_test_batches`
- Deprecated `ModelCheckpoint`'s attributes `best` and `kth_best_model` ([#1799](https://github.com/PyTorchLightning/pytorch-lightning/pull/1799))
- Dropped official support/testing for older PyTorch versions <1.3 ([#1917](https://github.com/PyTorchLightning/pytorch-lightning/pull/1917))
- Deprecated Trainer `proc_rank` in favour of `global_rank` ([#2166](https://github.com/PyTorchLightning/pytorch-lightning/pull/2166),
     [#2269](https://github.com/PyTorchLightning/pytorch-lightning/pull/2269))

### Removed

- Removed unintended Trainer argument `progress_bar_callback`, the callback should be passed in by `Trainer(callbacks=[...])` instead ([#1855](https://github.com/PyTorchLightning/pytorch-lightning/pull/1855))
- Removed obsolete `self._device` in Trainer ([#1849](https://github.com/PyTorchLightning/pytorch-lightning/pull/1849))
- Removed deprecated API ([#2073](https://github.com/PyTorchLightning/pytorch-lightning/pull/2073))
   * Packages: `pytorch_lightning.pt_overrides`, `pytorch_lightning.root_module`
   * Modules: `pytorch_lightning.logging.comet_logger`, `pytorch_lightning.logging.mlflow_logger`, `pytorch_lightning.logging.test_tube_logger`, `pytorch_lightning.overrides.override_data_parallel`, `pytorch_lightning.core.model_saving`, `pytorch_lightning.core.root_module`
   * Trainer arguments: `add_row_log_interval`, `default_save_path`, `gradient_clip`, `nb_gpu_nodes`, `max_nb_epochs`, `min_nb_epochs`, `nb_sanity_val_steps`
   * Trainer attributes: `nb_gpu_nodes`, `num_gpu_nodes`, `gradient_clip`, `max_nb_epochs`, `min_nb_epochs`, `nb_sanity_val_steps`, `default_save_path`, `tng_tqdm_dic`

### Fixed

- Run graceful training teardown on interpreter exit ([#1631](https://github.com/PyTorchLightning/pytorch-lightning/pull/1631))
- Fixed user warning when apex was used together with learning rate schedulers ([#1873](https://github.com/PyTorchLightning/pytorch-lightning/pull/1873))
- Fixed multiple calls of `EarlyStopping` callback ([#1863](https://github.com/PyTorchLightning/pytorch-lightning/pull/1863))
- Fixed an issue with `Trainer.from_argparse_args` when passing in unknown Trainer args ([#1932](https://github.com/PyTorchLightning/pytorch-lightning/pull/1932))
- Fixed bug related to logger not being reset correctly for model after tuner algorithms ([#1933](https://github.com/PyTorchLightning/pytorch-lightning/pull/1933))
- Fixed root node resolution for SLURM cluster with dash in host name ([#1954](https://github.com/PyTorchLightning/pytorch-lightning/pull/1954))
- Fixed `LearningRateLogger` in multi-scheduler setting ([#1944](https://github.com/PyTorchLightning/pytorch-lightning/pull/1944))
- Fixed test configuration check and testing ([#1804](https://github.com/PyTorchLightning/pytorch-lightning/pull/1804))
- Fixed an issue with Trainer constructor silently ignoring unknown/misspelled arguments ([#1820](https://github.com/PyTorchLightning/pytorch-lightning/pull/1820))
- Fixed `save_weights_only` in ModelCheckpoint ([#1780](https://github.com/PyTorchLightning/pytorch-lightning/pull/1780))
- Allow use of same `WandbLogger` instance for multiple training loops ([#2055](https://github.com/PyTorchLightning/pytorch-lightning/pull/2055))
- Fixed an issue with `_auto_collect_arguments` collecting local variables that are not constructor arguments and not working for signatures that have the instance not named `self` ([#2048](https://github.com/PyTorchLightning/pytorch-lightning/pull/2048))
- Fixed mistake in parameters' grad norm tracking ([#2012](https://github.com/PyTorchLightning/pytorch-lightning/pull/2012))
- Fixed CPU and hanging GPU crash ([#2118](https://github.com/PyTorchLightning/pytorch-lightning/pull/2118))
- Fixed an issue with the model summary and `example_input_array` depending on a specific ordering of the submodules in a LightningModule ([#1773](https://github.com/PyTorchLightning/pytorch-lightning/pull/1773))
- Fixed Tpu logging ([#2230](https://github.com/PyTorchLightning/pytorch-lightning/pull/2230))
- Fixed Pid port + duplicate `rank_zero` logging ([#2140](https://github.com/PyTorchLightning/pytorch-lightning/pull/2140),
     [#2231](https://github.com/PyTorchLightning/pytorch-lightning/pull/2231))

## [0.7.6] - 2020-05-16

### Added

- Added callback for logging learning rates ([#1498](https://github.com/PyTorchLightning/pytorch-lightning/pull/1498))
- Added transfer learning example (for a binary classification task in computer vision) ([#1564](https://github.com/PyTorchLightning/pytorch-lightning/pull/1564))
- Added type hints in `Trainer.fit()` and `Trainer.test()` to reflect that also a list of dataloaders can be passed in ([#1723](https://github.com/PyTorchLightning/pytorch-lightning/pull/1723)).
- Added auto scaling of batch size ([#1638](https://github.com/PyTorchLightning/pytorch-lightning/pull/1638))
- The progress bar metrics now also get updated in `training_epoch_end` ([#1724](https://github.com/PyTorchLightning/pytorch-lightning/pull/1724))
- Enable `NeptuneLogger` to work with `distributed_backend=ddp` ([#1753](https://github.com/PyTorchLightning/pytorch-lightning/pull/1753))
- Added option to provide seed to random generators to ensure reproducibility ([#1572](https://github.com/PyTorchLightning/pytorch-lightning/pull/1572))
- Added override for hparams in `load_from_ckpt` ([#1797](https://github.com/PyTorchLightning/pytorch-lightning/pull/1797))
- Added support multi-node distributed execution under `torchelastic` ([#1811](https://github.com/PyTorchLightning/pytorch-lightning/pull/1811),
     [#1818](https://github.com/PyTorchLightning/pytorch-lightning/pull/1818))
- Added using `store_true` for bool args ([#1822](https://github.com/PyTorchLightning/pytorch-lightning/pull/1822),
     [#1842](https://github.com/PyTorchLightning/pytorch-lightning/pull/1842))
- Added dummy logger for internally disabling logging for some features ([#1836](https://github.com/PyTorchLightning/pytorch-lightning/pull/1836))

### Changed

- Enable `non-blocking` for device transfers to GPU ([#1843](https://github.com/PyTorchLightning/pytorch-lightning/pull/1843))
- Replace mata_tags.csv with hparams.yaml ([#1271](https://github.com/PyTorchLightning/pytorch-lightning/pull/1271))
- Reduction when `batch_size < num_gpus` ([#1609](https://github.com/PyTorchLightning/pytorch-lightning/pull/1609))
- Updated LightningTemplateModel to look more like Colab example ([#1577](https://github.com/PyTorchLightning/pytorch-lightning/pull/1577))
- Don't convert `namedtuple` to `tuple` when transferring the batch to target device ([#1589](https://github.com/PyTorchLightning/pytorch-lightning/pull/1589))
- Allow passing hparams as keyword argument to LightningModule when loading from checkpoint ([#1639](https://github.com/PyTorchLightning/pytorch-lightning/pull/1639))
- Args should come after the last positional argument ([#1807](https://github.com/PyTorchLightning/pytorch-lightning/pull/1807))
- Made ddp the default if no backend specified with multiple GPUs ([#1789](https://github.com/PyTorchLightning/pytorch-lightning/pull/1789))

### Deprecated

- Deprecated `tags_csv` in favor of `hparams_file` ([#1271](https://github.com/PyTorchLightning/pytorch-lightning/pull/1271))

### Fixed

- Fixed broken link in PR template ([#1675](https://github.com/PyTorchLightning/pytorch-lightning/pull/1675))
- Fixed ModelCheckpoint not None checking filepath ([#1654](https://github.com/PyTorchLightning/pytorch-lightning/pull/1654))
- Trainer now calls `on_load_checkpoint()` when resuming from a checkpoint ([#1666](https://github.com/PyTorchLightning/pytorch-lightning/pull/1666))
- Fixed sampler logic for ddp with iterable dataset ([#1734](https://github.com/PyTorchLightning/pytorch-lightning/pull/1734))
- Fixed `_reset_eval_dataloader()` for IterableDataset ([#1560](https://github.com/PyTorchLightning/pytorch-lightning/pull/1560))
- Fixed Horovod distributed backend to set the `root_gpu` property ([#1669](https://github.com/PyTorchLightning/pytorch-lightning/pull/1669))
- Fixed wandb logger `global_step` affects other loggers ([#1492](https://github.com/PyTorchLightning/pytorch-lightning/pull/1492))
- Fixed disabling progress bar on non-zero ranks using Horovod backend ([#1709](https://github.com/PyTorchLightning/pytorch-lightning/pull/1709))
- Fixed bugs that prevent lr finder to be used together with early stopping and validation dataloaders ([#1676](https://github.com/PyTorchLightning/pytorch-lightning/pull/1676))
- Fixed a bug in Trainer that prepended the checkpoint path with `version_` when it shouldn't ([#1748](https://github.com/PyTorchLightning/pytorch-lightning/pull/1748))
- Fixed lr key name in case of param groups in LearningRateLogger ([#1719](https://github.com/PyTorchLightning/pytorch-lightning/pull/1719))
- Fixed saving native AMP scaler state (introduced in [#1561](https://github.com/PyTorchLightning/pytorch-lightning/pull/1561))
- Fixed accumulation parameter and suggestion method for learning rate finder ([#1801](https://github.com/PyTorchLightning/pytorch-lightning/pull/1801))
- Fixed num processes wasn't being set properly and auto sampler was ddp failing ([#1819](https://github.com/PyTorchLightning/pytorch-lightning/pull/1819))
- Fixed bugs in semantic segmentation example ([#1824](https://github.com/PyTorchLightning/pytorch-lightning/pull/1824))
- Fixed saving native AMP scaler state ([#1561](https://github.com/PyTorchLightning/pytorch-lightning/pull/1561),
     [#1777](https://github.com/PyTorchLightning/pytorch-lightning/pull/1777))
- Fixed native amp + ddp ([#1788](https://github.com/PyTorchLightning/pytorch-lightning/pull/1788))
- Fixed `hparam` logging with metrics ([#1647](https://github.com/PyTorchLightning/pytorch-lightning/pull/1647))

## [0.7.5] - 2020-04-27

### Changed

- Allow logging of metrics together with `hparams` ([#1630](https://github.com/PyTorchLightning/pytorch-lightning/pull/1630))
- Allow metrics logged together with hparams ([#1630](https://github.com/PyTorchLightning/pytorch-lightning/pull/1630))

### Removed

- Removed Warning from trainer loop ([#1634](https://github.com/PyTorchLightning/pytorch-lightning/pull/1634))

### Fixed

- Fixed ModelCheckpoint not being fixable ([#1632](https://github.com/PyTorchLightning/pytorch-lightning/pull/1632))
- Fixed CPU DDP breaking change and DDP change ([#1635](https://github.com/PyTorchLightning/pytorch-lightning/pull/1635))
- Tested pickling ([#1636](https://github.com/PyTorchLightning/pytorch-lightning/pull/1636))


## [0.7.4] - 2020-04-26

### Added

- Added flag `replace_sampler_ddp` to manually disable sampler replacement in DDP  ([#1513](https://github.com/PyTorchLightning/pytorch-lightning/pull/1513))
- Added speed parity tests (max 1 sec difference per epoch)([#1482](https://github.com/PyTorchLightning/pytorch-lightning/pull/1482))
- Added `auto_select_gpus` flag to trainer that enables automatic selection of available GPUs on exclusive mode systems.
- Added learning rate finder ([#1347](https://github.com/PyTorchLightning/pytorch-lightning/pull/1347))
- Added support for ddp mode in clusters without SLURM ([#1387](https://github.com/PyTorchLightning/pytorch-lightning/pull/1387))
- Added `test_dataloaders` parameter to `Trainer.test()` ([#1434](https://github.com/PyTorchLightning/pytorch-lightning/pull/1434))
- Added `terminate_on_nan` flag to trainer that performs a NaN check with each training iteration when set to `True` ([#1475](https://github.com/PyTorchLightning/pytorch-lightning/pull/1475))
- Added speed parity tests (max 1 sec difference per epoch)([#1482](https://github.com/PyTorchLightning/pytorch-lightning/pull/1482))
- Added `terminate_on_nan` flag to trainer that performs a NaN check with each training iteration when set to `True`. ([#1475](https://github.com/PyTorchLightning/pytorch-lightning/pull/1475))
- Added `ddp_cpu` backend for testing ddp without GPUs ([#1158](https://github.com/PyTorchLightning/pytorch-lightning/pull/1158))
- Added [Horovod](http://horovod.ai) support as a distributed backend `Trainer(distributed_backend='horovod')` ([#1529](https://github.com/PyTorchLightning/pytorch-lightning/pull/1529))
- Added support for 8 core distributed training on Kaggle TPU's ([#1568](https://github.com/PyTorchLightning/pytorch-lightning/pull/1568))
- Added support for native AMP ([#1561](https://github.com/PyTorchLightning/pytorch-lightning/pull/1561),
    [#1580](https://github.com/PyTorchLightning/pytorch-lightning/pull/1580))

### Changed

- Changed the default behaviour to no longer include a NaN check with each training iteration. ([#1475](https://github.com/PyTorchLightning/pytorch-lightning/pull/1475))
- Decoupled the progress bar from trainer` it is a callback now and can be customized or even be replaced entirely ([#1450](https://github.com/PyTorchLightning/pytorch-lightning/pull/1450)).
- Changed lr schedule step interval behavior to update every backwards pass instead of every forwards pass ([#1477](https://github.com/PyTorchLightning/pytorch-lightning/pull/1477))
- Defines shared proc. rank, remove rank from instances (e.g. loggers) ([#1408](https://github.com/PyTorchLightning/pytorch-lightning/pull/1408))
- Updated semantic segmentation example with custom U-Net and logging ([#1371](https://github.com/PyTorchLightning/pytorch-lightning/pull/1371))
- Disabled val and test shuffling ([#1600](https://github.com/PyTorchLightning/pytorch-lightning/pull/1600))

### Deprecated

- Deprecated `training_tqdm_dict` in favor of `progress_bar_dict` ([#1450](https://github.com/PyTorchLightning/pytorch-lightning/pull/1450)).

### Removed

- Removed `test_dataloaders` parameter from `Trainer.fit()` ([#1434](https://github.com/PyTorchLightning/pytorch-lightning/pull/1434))

### Fixed

- Added the possibility to pass nested metrics dictionaries to loggers ([#1582](https://github.com/PyTorchLightning/pytorch-lightning/pull/1582))
- Fixed memory leak from opt return ([#1528](https://github.com/PyTorchLightning/pytorch-lightning/pull/1528))
- Fixed saving checkpoint before deleting old ones ([#1453](https://github.com/PyTorchLightning/pytorch-lightning/pull/1453))
- Fixed loggers - flushing last logged metrics even before continue, e.g. `trainer.test()` results ([#1459](https://github.com/PyTorchLightning/pytorch-lightning/pull/1459))
- Fixed optimizer configuration when `configure_optimizers` returns dict without `lr_scheduler` ([#1443](https://github.com/PyTorchLightning/pytorch-lightning/pull/1443))
- Fixed `LightningModule` - mixing hparams and arguments in `LightningModule.__init__()` crashes load_from_checkpoint() ([#1505](https://github.com/PyTorchLightning/pytorch-lightning/pull/1505))
- Added a missing call to the `on_before_zero_grad` model hook ([#1493](https://github.com/PyTorchLightning/pytorch-lightning/pull/1493)).
- Allow use of sweeps with `WandbLogger` ([#1512](https://github.com/PyTorchLightning/pytorch-lightning/pull/1512))
- Fixed a bug that caused the `callbacks` Trainer argument to reference a global variable ([#1534](https://github.com/PyTorchLightning/pytorch-lightning/pull/1534)).
- Fixed a bug that set all boolean CLI arguments from `Trainer.add_argparse_args` always to True ([#1571](https://github.com/PyTorchLightning/pytorch-lightning/pull/1571))
- Fixed do not copy the batch when training on a single GPU ([#1576](https://github.com/PyTorchLightning/pytorch-lightning/pull/1576),
    [#1579](https://github.com/PyTorchLightning/pytorch-lightning/pull/1579))
- Fixed soft checkpoint removing on DDP ([#1408](https://github.com/PyTorchLightning/pytorch-lightning/pull/1408))
- Fixed automatic parser bug ([#1585](https://github.com/PyTorchLightning/pytorch-lightning/pull/1585))
- Fixed bool conversion from string ([#1606](https://github.com/PyTorchLightning/pytorch-lightning/pull/1606))

## [0.7.3] - 2020-04-09

### Added

- Added `rank_zero_warn` for warning only in rank 0 ([#1428](https://github.com/PyTorchLightning/pytorch-lightning/pull/1428))

### Fixed

- Fixed default `DistributedSampler` for DDP training ([#1425](https://github.com/PyTorchLightning/pytorch-lightning/pull/1425))
- Fixed workers warning not on windows ([#1430](https://github.com/PyTorchLightning/pytorch-lightning/pull/1430))
- Fixed returning tuple from `run_training_batch` ([#1431](https://github.com/PyTorchLightning/pytorch-lightning/pull/1431))
- Fixed gradient clipping ([#1438](https://github.com/PyTorchLightning/pytorch-lightning/pull/1438))
- Fixed pretty print ([#1441](https://github.com/PyTorchLightning/pytorch-lightning/pull/1441))


## [0.7.2] - 2020-04-07

### Added

- Added same step loggers' metrics aggregation ([#1278](https://github.com/PyTorchLightning/pytorch-lightning/pull/1278))
- Added parity test between a vanilla MNIST model and lightning model ([#1284](https://github.com/PyTorchLightning/pytorch-lightning/pull/1284))
- Added parity test between a vanilla RNN model and lightning model ([#1351](https://github.com/PyTorchLightning/pytorch-lightning/pull/1351))
- Added Reinforcement Learning - Deep Q-network (DQN) lightning example ([#1232](https://github.com/PyTorchLightning/pytorch-lightning/pull/1232))
- Added support for hierarchical `dict` ([#1152](https://github.com/PyTorchLightning/pytorch-lightning/pull/1152))
- Added `TrainsLogger` class ([#1122](https://github.com/PyTorchLightning/pytorch-lightning/pull/1122))
- Added type hints to `pytorch_lightning.core` ([#946](https://github.com/PyTorchLightning/pytorch-lightning/pull/946))
- Added support for `IterableDataset` in validation and testing ([#1104](https://github.com/PyTorchLightning/pytorch-lightning/pull/1104))
- Added support for non-primitive types in `hparams` for `TensorboardLogger` ([#1130](https://github.com/PyTorchLightning/pytorch-lightning/pull/1130))
- Added a check that stops the training when loss or weights contain `NaN` or `inf` values. ([#1097](https://github.com/PyTorchLightning/pytorch-lightning/pull/1097))
- Added support for `IterableDataset` when `val_check_interval=1.0` (default), this will trigger validation at the end of each epoch. ([#1283](https://github.com/PyTorchLightning/pytorch-lightning/pull/1283))
- Added `summary` method to Profilers. ([#1259](https://github.com/PyTorchLightning/pytorch-lightning/pull/1259))
- Added informative errors if user defined dataloader has zero length ([#1280](https://github.com/PyTorchLightning/pytorch-lightning/pull/1280))
- Added testing for python 3.8 ([#915](https://github.com/PyTorchLightning/pytorch-lightning/pull/915))
- Added a `training_epoch_end` method which is the mirror of `validation_epoch_end`. ([#1357](https://github.com/PyTorchLightning/pytorch-lightning/pull/1357))
- Added model configuration checking ([#1199](https://github.com/PyTorchLightning/pytorch-lightning/pull/1199))
- Added support for optimizer frequencies through `LightningModule.configure_optimizers()` ([#1269](https://github.com/PyTorchLightning/pytorch-lightning/pull/1269))
- Added option to run without an optimizer by returning `None` from `configure_optimizers`. ([#1279](https://github.com/PyTorchLightning/pytorch-lightning/pull/1279))
- Added a warning when the number of data loader workers is small. ([#1378](https://github.com/PyTorchLightning/pytorch-lightning/pull/1378))

### Changed

- Changed (renamed and refatored) `TensorRunningMean` -> `TensorRunningAccum`: running accumulations were generalized. ([#1278](https://github.com/PyTorchLightning/pytorch-lightning/pull/1278))
- Changed `progress_bar_refresh_rate` trainer flag to disable progress bar when set to 0. ([#1108](https://github.com/PyTorchLightning/pytorch-lightning/pull/1108))
- Enhanced `load_from_checkpoint` to also forward params to the model ([#1307](https://github.com/PyTorchLightning/pytorch-lightning/pull/1307))
- Updated references to `self.forward()` to instead use the `__call__` interface. ([#1211](https://github.com/PyTorchLightning/pytorch-lightning/pull/1211))
- Changed default behaviour of `configure_optimizers` to use no optimizer rather than Adam. ([#1279](https://github.com/PyTorchLightning/pytorch-lightning/pull/1279))
- Allow to upload models on W&B ([#1339](https://github.com/PyTorchLightning/pytorch-lightning/pull/1339))
- On DP and DDP2 unsqueeze is automated now ([#1319](https://github.com/PyTorchLightning/pytorch-lightning/pull/1319))
- Did not always create a DataLoader during reinstantiation, but the same type as before (if subclass of DataLoader) ([#1346](https://github.com/PyTorchLightning/pytorch-lightning/pull/1346))
- Did not interfere with a default sampler ([#1318](https://github.com/PyTorchLightning/pytorch-lightning/pull/1318))
- Remove default Adam optimizer ([#1317](https://github.com/PyTorchLightning/pytorch-lightning/pull/1317))
- Give warnings for unimplemented required lightning methods ([#1317](https://github.com/PyTorchLightning/pytorch-lightning/pull/1317))
- Made `evaluate` method private >> `Trainer._evaluate(...)`. ([#1260](https://github.com/PyTorchLightning/pytorch-lightning/pull/1260))
- Simplify the PL examples structure (shallower and more readable) ([#1247](https://github.com/PyTorchLightning/pytorch-lightning/pull/1247))
- Changed min max gpu memory to be on their own plots ([#1358](https://github.com/PyTorchLightning/pytorch-lightning/pull/1358))
- Remove `.item` which causes sync issues ([#1254](https://github.com/PyTorchLightning/pytorch-lightning/pull/1254))
- Changed smoothing in TQDM to decrease variability of time remaining between training / eval ([#1194](https://github.com/PyTorchLightning/pytorch-lightning/pull/1194))
- Change default logger to dedicated one ([#1064](https://github.com/PyTorchLightning/pytorch-lightning/pull/1064))

### Deprecated

- Deprecated Trainer argument `print_nan_grads` ([#1097](https://github.com/PyTorchLightning/pytorch-lightning/pull/1097))
- Deprecated Trainer argument `show_progress_bar` ([#1108](https://github.com/PyTorchLightning/pytorch-lightning/pull/1108))

### Removed

- Removed test for no test dataloader in .fit ([#1495](https://github.com/PyTorchLightning/pytorch-lightning/pull/1495))
- Removed duplicated module `pytorch_lightning.utilities.arg_parse` for loading CLI arguments ([#1167](https://github.com/PyTorchLightning/pytorch-lightning/pull/1167))
- Removed wandb logger's `finalize` method ([#1193](https://github.com/PyTorchLightning/pytorch-lightning/pull/1193))
- Dropped `torchvision` dependency in tests and added own MNIST dataset class instead ([#986](https://github.com/PyTorchLightning/pytorch-lightning/pull/986))

### Fixed

- Fixed `model_checkpoint` when saving all models ([#1359](https://github.com/PyTorchLightning/pytorch-lightning/pull/1359))
- `Trainer.add_argparse_args` classmethod fixed. Now it adds a type for the arguments ([#1147](https://github.com/PyTorchLightning/pytorch-lightning/pull/1147))
- Fixed bug related to type checking of `ReduceLROnPlateau` lr schedulers([#1126](https://github.com/PyTorchLightning/pytorch-lightning/pull/1126))
- Fixed a bug to ensure lightning checkpoints to be backward compatible ([#1132](https://github.com/PyTorchLightning/pytorch-lightning/pull/1132))
- Fixed a bug that created an extra dataloader with active `reload_dataloaders_every_epoch` ([#1196](https://github.com/PyTorchLightning/pytorch-lightning/pull/1196))
- Fixed all warnings and errors in the docs build process ([#1191](https://github.com/PyTorchLightning/pytorch-lightning/pull/1191))
- Fixed an issue where `val_percent_check=0` would not disable validation ([#1251](https://github.com/PyTorchLightning/pytorch-lightning/pull/1251))
- Fixed average of incomplete `TensorRunningMean` ([#1309](https://github.com/PyTorchLightning/pytorch-lightning/pull/1309))
- Fixed `WandbLogger.watch` with `wandb.init()` ([#1311](https://github.com/PyTorchLightning/pytorch-lightning/pull/1311))
- Fixed an issue with early stopping that would prevent it from monitoring training metrics when validation is disabled / not implemented ([#1235](https://github.com/PyTorchLightning/pytorch-lightning/pull/1235)).
- Fixed a bug that would cause `trainer.test()` to run on the validation set when overloading `validation_epoch_end` and `test_end` ([#1353](https://github.com/PyTorchLightning/pytorch-lightning/pull/1353))
- Fixed `WandbLogger.watch` - use of the watch method without importing `wandb` ([#1311](https://github.com/PyTorchLightning/pytorch-lightning/pull/1311))
- Fixed `WandbLogger` to be used with 'ddp' - allow reinits in sub-processes ([#1149](https://github.com/PyTorchLightning/pytorch-lightning/pull/1149),
     [#1360](https://github.com/PyTorchLightning/pytorch-lightning/pull/1360))
- Made `training_epoch_end` behave like `validation_epoch_end` ([#1357](https://github.com/PyTorchLightning/pytorch-lightning/pull/1357))
- Fixed `fast_dev_run` running validation twice ([#1365](https://github.com/PyTorchLightning/pytorch-lightning/pull/1365))
- Fixed pickle error from quick patch `__code__` ([#1352](https://github.com/PyTorchLightning/pytorch-lightning/pull/1352))
- Fixed memory leak on GPU0 ([#1094](https://github.com/PyTorchLightning/pytorch-lightning/pull/1094),
     [#1349](https://github.com/PyTorchLightning/pytorch-lightning/pull/1349))
- Fixed checkpointing interval ([#1272](https://github.com/PyTorchLightning/pytorch-lightning/pull/1272))
- Fixed validation and training loops run the partial dataset ([#1192](https://github.com/PyTorchLightning/pytorch-lightning/pull/1192))
- Fixed running `on_validation_end` only on main process in DDP ([#1125](https://github.com/PyTorchLightning/pytorch-lightning/pull/1125))
- Fixed `load_spawn_weights` only in proc rank 0 ([#1385](https://github.com/PyTorchLightning/pytorch-lightning/pull/1385))
- Fixes `use_amp` issue ([#1145](https://github.com/PyTorchLightning/pytorch-lightning/pull/1145))
- Fixes using deprecated `use_amp` attribute ([#1145](https://github.com/PyTorchLightning/pytorch-lightning/pull/1145))
- Fixed Tensorboard logger error: lightning_logs directory not exists in multi-node DDP on nodes with rank != 0 ([#1377](https://github.com/PyTorchLightning/pytorch-lightning/pull/1377))
- Fixed `Unimplemented backend XLA` error on TPU ([#1387](https://github.com/PyTorchLightning/pytorch-lightning/pull/1387))

## [0.7.1] - 2020-03-07

### Fixed

- Fixes `print` issues and `data_loader` ([#1080](https://github.com/PyTorchLightning/pytorch-lightning/pull/1080))

## [0.7.0] - 2020-03-06

### Added

- Added automatic sampler setup. Depending on DDP or TPU, lightning configures the sampler correctly (user needs to do nothing) ([#926](https://github.com/PyTorchLightning/pytorch-lightning/pull/926))
- Added `reload_dataloaders_every_epoch=False` flag for trainer. Some users require reloading data every epoch ([#926](https://github.com/PyTorchLightning/pytorch-lightning/pull/926))
- Added `progress_bar_refresh_rate=50` flag for trainer. Throttle refresh rate on notebooks ([#926](https://github.com/PyTorchLightning/pytorch-lightning/pull/926))
- Updated governance docs
- Added a check to ensure that the metric used for early stopping exists before training commences ([#542](https://github.com/PyTorchLightning/pytorch-lightning/pull/542))
- Added `optimizer_idx` argument to `backward` hook ([#733](https://github.com/PyTorchLightning/pytorch-lightning/pull/733))
- Added `entity` argument to `WandbLogger` to be passed to `wandb.init` ([#783](https://github.com/PyTorchLightning/pytorch-lightning/pull/783))
- Added a tool for profiling training runs ([#782](https://github.com/PyTorchLightning/pytorch-lightning/pull/782))
- Improved flexibility for naming of TensorBoard logs, can now set `version` to a `str` to just save to that directory, and use `name=''` to prevent experiment-name directory ([#804](https://github.com/PyTorchLightning/pytorch-lightning/pull/804))
- Added option to specify `step` key when logging metrics ([#808](https://github.com/PyTorchLightning/pytorch-lightning/pull/808))
- Added `train_dataloader`, `val_dataloader` and `test_dataloader` arguments to `Trainer.fit()`, for alternative data parsing ([#759](https://github.com/PyTorchLightning/pytorch-lightning/pull/759))
- Added Tensor Processing Unit (TPU) support ([#868](https://github.com/PyTorchLightning/pytorch-lightning/pull/868))
- Added semantic segmentation example ([#751](https://github.com/PyTorchLightning/pytorch-lightning/pull/751),[#876](https://github.com/PyTorchLightning/pytorch-lightning/pull/876),
     [#881](https://github.com/PyTorchLightning/pytorch-lightning/pull/881))
- Split callbacks in multiple files ([#849](https://github.com/PyTorchLightning/pytorch-lightning/pull/849))
- Support for user defined callbacks ([#889](https://github.com/PyTorchLightning/pytorch-lightning/pull/889) and [#950](https://github.com/PyTorchLightning/pytorch-lightning/pull/950))
- Added support for multiple loggers to be passed to `Trainer` as an iterable (e.g. list, tuple, etc.) ([#903](https://github.com/PyTorchLightning/pytorch-lightning/pull/903))
- Added support for step-based learning rate scheduling ([#941](https://github.com/PyTorchLightning/pytorch-lightning/pull/941))
- Added support for logging `hparams` as dict ([#1029](https://github.com/PyTorchLightning/pytorch-lightning/pull/1029))
- Checkpoint and early stopping now work without val. step ([#1041](https://github.com/PyTorchLightning/pytorch-lightning/pull/1041))
- Support graceful training cleanup after Keyboard Interrupt ([#856](https://github.com/PyTorchLightning/pytorch-lightning/pull/856),
     [#1019](https://github.com/PyTorchLightning/pytorch-lightning/pull/1019))
- Added type hints for function arguments ([#912](https://github.com/PyTorchLightning/pytorch-lightning/pull/912), )
- Added default `argparser` for `Trainer` ([#952](https://github.com/PyTorchLightning/pytorch-lightning/pull/1023),
     [#1023](https://github.com/PyTorchLightning/pytorch-lightning/pull/1023))
- Added TPU gradient clipping ([#963](https://github.com/PyTorchLightning/pytorch-lightning/pull/963))
- Added max/min number of steps in `Trainer` ([#728](https://github.com/PyTorchLightning/pytorch-lightning/pull/728))

### Changed

- Improved `NeptuneLogger` by adding `close_after_fit` argument to allow logging after training([#908](https://github.com/PyTorchLightning/pytorch-lightning/pull/1084))
- Changed default TQDM to use `tqdm.auto` for prettier outputs in IPython notebooks ([#752](https://github.com/PyTorchLightning/pytorch-lightning/pull/752))
- Changed `pytorch_lightning.logging` to `pytorch_lightning.loggers` ([#767](https://github.com/PyTorchLightning/pytorch-lightning/pull/767))
- Moved the default `tqdm_dict` definition from Trainer to `LightningModule`, so it can be overridden by the user ([#749](https://github.com/PyTorchLightning/pytorch-lightning/pull/749))
- Moved functionality of `LightningModule.load_from_metrics` into `LightningModule.load_from_checkpoint` ([#995](https://github.com/PyTorchLightning/pytorch-lightning/pull/995))
- Changed Checkpoint path parameter from `filepath` to `dirpath` ([#1016](https://github.com/PyTorchLightning/pytorch-lightning/pull/1016))
- Freezed models `hparams` as `Namespace` property ([#1029](https://github.com/PyTorchLightning/pytorch-lightning/pull/1029))
- Dropped `logging` config in package init ([#1015](https://github.com/PyTorchLightning/pytorch-lightning/pull/1015))
- Renames model steps ([#1051](https://github.com/PyTorchLightning/pytorch-lightning/pull/1051))
  - `training_end` >> `training_epoch_end`
  - `validation_end` >> `validation_epoch_end`
  - `test_end` >> `test_epoch_end`
- Refactor dataloading, supports infinite dataloader ([#955](https://github.com/PyTorchLightning/pytorch-lightning/pull/955))
- Create single file in `TensorBoardLogger` ([#777](https://github.com/PyTorchLightning/pytorch-lightning/pull/777))

### Deprecated

- Deprecated `pytorch_lightning.logging` ([#767](https://github.com/PyTorchLightning/pytorch-lightning/pull/767))
- Deprecated `LightningModule.load_from_metrics` in favour of `LightningModule.load_from_checkpoint` ([#995](https://github.com/PyTorchLightning/pytorch-lightning/pull/995),
     [#1079](https://github.com/PyTorchLightning/pytorch-lightning/pull/1079))
- Deprecated `@data_loader` decorator ([#926](https://github.com/PyTorchLightning/pytorch-lightning/pull/926))
- Deprecated model steps `training_end`, `validation_end` and `test_end` ([#1051](https://github.com/PyTorchLightning/pytorch-lightning/pull/1051),
     [#1056](https://github.com/PyTorchLightning/pytorch-lightning/pull/1056))

### Removed

- Removed dependency on `pandas` ([#736](https://github.com/PyTorchLightning/pytorch-lightning/pull/736))
- Removed dependency on `torchvision` ([#797](https://github.com/PyTorchLightning/pytorch-lightning/pull/797))
- Removed dependency on `scikit-learn` ([#801](https://github.com/PyTorchLightning/pytorch-lightning/pull/801))

### Fixed

- Fixed a bug where early stopping `on_end_epoch` would be called inconsistently when `check_val_every_n_epoch == 0` ([#743](https://github.com/PyTorchLightning/pytorch-lightning/pull/743))
- Fixed a bug where the model checkpointer didn't write to the same directory as the logger ([#771](https://github.com/PyTorchLightning/pytorch-lightning/pull/771))
- Fixed a bug where the `TensorBoardLogger` class would create an additional empty log file during fitting ([#777](https://github.com/PyTorchLightning/pytorch-lightning/pull/777))
- Fixed a bug where `global_step` was advanced incorrectly when using `accumulate_grad_batches > 1` ([#832](https://github.com/PyTorchLightning/pytorch-lightning/pull/832))
- Fixed a bug when calling `self.logger.experiment` with multiple loggers ([#1009](https://github.com/PyTorchLightning/pytorch-lightning/pull/1009))
- Fixed a bug when calling `logger.append_tags` on a `NeptuneLogger` with a single tag ([#1009](https://github.com/PyTorchLightning/pytorch-lightning/pull/1009))
- Fixed sending back data from `.spawn` by saving and loading the trained model in/out of the process ([#1017](https://github.com/PyTorchLightning/pytorch-lightning/pull/1017)
- Fixed port collision on DDP ([#1010](https://github.com/PyTorchLightning/pytorch-lightning/pull/1010))
- Fixed/tested pass overrides ([#918](https://github.com/PyTorchLightning/pytorch-lightning/pull/918))
- Fixed comet logger to log after train ([#892](https://github.com/PyTorchLightning/pytorch-lightning/pull/892))
- Remove deprecated args to learning rate step function ([#890](https://github.com/PyTorchLightning/pytorch-lightning/pull/890))

## [0.6.0] - 2020-01-21

### Added

- Added support for resuming from a specific checkpoint via `resume_from_checkpoint` argument ([#516](https://github.com/PyTorchLightning/pytorch-lightning/pull/516))
- Added support for `ReduceLROnPlateau` scheduler ([#320](https://github.com/PyTorchLightning/pytorch-lightning/pull/320))
- Added support for Apex mode `O2` in conjunction with Data Parallel ([#493](https://github.com/PyTorchLightning/pytorch-lightning/pull/493))
- Added option (`save_top_k`) to save the top k models in the `ModelCheckpoint` class ([#128](https://github.com/PyTorchLightning/pytorch-lightning/pull/128))
- Added `on_train_start` and `on_train_end` hooks to `ModelHooks` ([#598](https://github.com/PyTorchLightning/pytorch-lightning/pull/598))
- Added `TensorBoardLogger` ([#607](https://github.com/PyTorchLightning/pytorch-lightning/pull/607))
- Added support for weight summary of model with multiple inputs ([#543](https://github.com/PyTorchLightning/pytorch-lightning/pull/543))
- Added `map_location` argument to `load_from_metrics` and `load_from_checkpoint` ([#625](https://github.com/PyTorchLightning/pytorch-lightning/pull/625))
- Added option to disable validation by setting `val_percent_check=0` ([#649](https://github.com/PyTorchLightning/pytorch-lightning/pull/649))
- Added `NeptuneLogger` class ([#648](https://github.com/PyTorchLightning/pytorch-lightning/pull/648))
- Added `WandbLogger` class ([#627](https://github.com/PyTorchLightning/pytorch-lightning/pull/627))

### Changed

- Changed the default progress bar to print to stdout instead of stderr ([#531](https://github.com/PyTorchLightning/pytorch-lightning/pull/531))
- Renamed `step_idx` to `step`, `epoch_idx` to `epoch`, `max_num_epochs` to `max_epochs` and `min_num_epochs` to `min_epochs` ([#589](https://github.com/PyTorchLightning/pytorch-lightning/pull/589))
- Renamed `total_batch_nb` to `total_batches`, `nb_val_batches` to `num_val_batches`, `nb_training_batches` to `num_training_batches`, `max_nb_epochs` to `max_epochs`, `min_nb_epochs` to `min_epochs`, `nb_test_batches` to `num_test_batches`, and `nb_val_batches` to `num_val_batches` ([#567](https://github.com/PyTorchLightning/pytorch-lightning/pull/567))
- Changed gradient logging to use parameter names instead of indexes ([#660](https://github.com/PyTorchLightning/pytorch-lightning/pull/660))
- Changed the default logger to `TensorBoardLogger` ([#609](https://github.com/PyTorchLightning/pytorch-lightning/pull/609))
- Changed the directory for tensorboard logging to be the same as model checkpointing ([#706](https://github.com/PyTorchLightning/pytorch-lightning/pull/706))

### Deprecated

- Deprecated `max_nb_epochs` and `min_nb_epochs` ([#567](https://github.com/PyTorchLightning/pytorch-lightning/pull/567))
- Deprecated the `on_sanity_check_start` hook in `ModelHooks` ([#598](https://github.com/PyTorchLightning/pytorch-lightning/pull/598))

### Removed

- Removed the `save_best_only` argument from `ModelCheckpoint`, use `save_top_k=1` instead ([#128](https://github.com/PyTorchLightning/pytorch-lightning/pull/128))

### Fixed

- Fixed a bug which ocurred when using Adagrad with cuda ([#554](https://github.com/PyTorchLightning/pytorch-lightning/pull/554))
- Fixed a bug where training would be on the GPU despite setting `gpus=0` or `gpus=[]` ([#561](https://github.com/PyTorchLightning/pytorch-lightning/pull/561))
- Fixed an error with `print_nan_gradients` when some parameters do not require gradient ([#579](https://github.com/PyTorchLightning/pytorch-lightning/pull/579))
- Fixed a bug where the progress bar would show an incorrect number of total steps during the validation sanity check when using multiple validation data loaders ([#597](https://github.com/PyTorchLightning/pytorch-lightning/pull/597))
- Fixed support for PyTorch 1.1.0 ([#552](https://github.com/PyTorchLightning/pytorch-lightning/pull/552))
- Fixed an issue with early stopping when using a `val_check_interval < 1.0` in `Trainer` ([#492](https://github.com/PyTorchLightning/pytorch-lightning/pull/492))
- Fixed bugs relating to the `CometLogger` object that would cause it to not work properly ([#481](https://github.com/PyTorchLightning/pytorch-lightning/pull/481))
- Fixed a bug that would occur when returning `-1` from `on_batch_start` following an early exit or when the batch was `None` ([#509](https://github.com/PyTorchLightning/pytorch-lightning/pull/509))
- Fixed a potential race condition with several processes trying to create checkpoint directories ([#530](https://github.com/PyTorchLightning/pytorch-lightning/pull/530))
- Fixed a bug where batch 'segments' would remain on the GPU when using `truncated_bptt > 1` ([#532](https://github.com/PyTorchLightning/pytorch-lightning/pull/532))
- Fixed a bug when using `IterableDataset` ([#547](https://github.com/PyTorchLightning/pytorch-lightning/pull/547))
- Fixed a bug where `.item` was called on non-tensor objects ([#602](https://github.com/PyTorchLightning/pytorch-lightning/pull/602))
- Fixed a bug where `Trainer.train` would crash on an uninitialized variable if the trainer was run after resuming from a checkpoint that was already at `max_epochs` ([#608](https://github.com/PyTorchLightning/pytorch-lightning/pull/608))
- Fixed a bug where early stopping would begin two epochs early ([#617](https://github.com/PyTorchLightning/pytorch-lightning/pull/617))
- Fixed a bug where `num_training_batches` and `num_test_batches` would sometimes be rounded down to zero ([#649](https://github.com/PyTorchLightning/pytorch-lightning/pull/649))
- Fixed a bug where an additional batch would be processed when manually setting `num_training_batches` ([#653](https://github.com/PyTorchLightning/pytorch-lightning/pull/653))
- Fixed a bug when batches did not have a `.copy` method ([#701](https://github.com/PyTorchLightning/pytorch-lightning/pull/701))
- Fixed a bug when using `log_gpu_memory=True` in Python 3.6 ([#715](https://github.com/PyTorchLightning/pytorch-lightning/pull/715))
- Fixed a bug where checkpoint writing could exit before completion, giving incomplete checkpoints ([#689](https://github.com/PyTorchLightning/pytorch-lightning/pull/689))
- Fixed a bug where `on_train_end` was not called when ealy stopping ([#723](https://github.com/PyTorchLightning/pytorch-lightning/pull/723))

## [0.5.3] - 2019-11-06

### Added

- Added option to disable default logger, checkpointer, and early stopping by passing `logger=False`, `checkpoint_callback=False` and `early_stop_callback=False` respectively
- Added `CometLogger` for use with Comet.ml
- Added `val_check_interval` argument to `Trainer` allowing validition to be performed at every given number of batches
- Added functionality to save and load hyperparameters using the standard checkpoint mechanism
- Added call to `torch.cuda.empty_cache` before training starts
- Added option for user to override the call t `backward`
- Added support for truncated backprop through time via the `truncated_bptt_steps` argument in `Trainer`
- Added option to operate on all outputs from `training_step` in DDP2
- Added a hook for modifying DDP init
- Added a hook for modifying Apex

### Changed

- Changed experiment version to be padded with zeros (e.g. `/dir/version_9` becomes `/dir/version_0009`)
- Changed callback metrics to include any metrics given in logs or progress bar
- Changed the default for `save_best_only` in `ModelCheckpoint` to `True`
- Added `tng_data_loader` for backwards compatibility
- Renamed `MLFlowLogger.client` to `MLFlowLogger.experiment` for consistency
- Moved `global_step` increment to happen after the batch has been processed
- Changed weights restore to first attempt HPC weights before restoring normally, preventing both weights being restored and running out of memory
- Changed progress bar functionality to add multiple progress bars for train/val/test
- Changed calls to `print` to use `logging` instead

### Deprecated

- Deprecated `tng_dataloader`

### Fixed

- Fixed an issue where the number of batches was off by one during training
- Fixed a bug that occured when setting a ckeckpoint callback and `early_stop_callback=False`
- Fixed an error when importing CometLogger
- Fixed a bug where the `gpus` argument had some unexpected behaviour
- Fixed a bug where the computed total number of batches was sometimes incorrect
- Fixed a bug where the progress bar would sometimes not show the total number of batches in test mode
- Fixed a bug when using the `log_gpu_memory='min_max'` option in `Trainer`
- Fixed a bug where checkpointing would sometimes erase the current directory

## [0.5.2] - 2019-10-10

### Added

- Added `weights_summary` argument to `Trainer` to be set to `full` (full summary), `top` (just top level modules) or other
- Added `tags` argument to `MLFlowLogger`

### Changed

- Changed default for `amp_level` to `O1`

### Removed

- Removed the `print_weights_summary` argument from `Trainer`

### Fixed

- Fixed a bug where logs were not written properly
- Fixed a bug where `logger.finalize` wasn't called after training is complete
- Fixed callback metric errors in DDP
- Fixed a bug where `TestTubeLogger` didn't log to the correct directory

## [0.5.1] - 2019-10-05

### Added

- Added the `LightningLoggerBase` class for experiment loggers
- Added `MLFlowLogger` for logging with `mlflow`
- Added `TestTubeLogger` for logging with `test_tube`
- Added a different implementation of DDP (`distributed_backed='ddp2'`) where every node has one model using all GPUs
- Added support for optimisers which require a closure (e.g. LBFGS)
- Added automatic `MASTER_PORT` defualt for DDP when not set manually
- Added new GPU memory logging options `'min_max'` (log only the min/max utilization) and `'all'` (log all the GPU memory)

### Changed

- Changed schedulers to always be called with the current epoch
- Changed `test_tube` to an optional dependency
- Changed data loaders to internally use a getter instead of a python property
- Disabled auto GPU loading when restoring weights to prevent out of memory errors
- Changed logging, early stopping and checkpointing to occur by default

### Fixed

- Fixed a bug with samplers that do not specify `set_epoch`
- Fixed a bug when using the `MLFlowLogger` with unsupported data types, this will now raise a warning
- Fixed a bug where gradient norms were alwasy zero using `track_grad_norm`
- Fixed a bug which causes a crash when logging memory

## [0.5.0] - 2019-09-26

### Changed

- Changed `data_batch` argument to `batch` throughout
- Changed `batch_i` argument to `batch_idx` throughout
- Changed `tng_dataloader` method to `train_dataloader`
- Changed `on_tng_metrics` method to `on_training_metrics`
- Changed `gradient_clip` argument to `gradient_clip_val`
- Changed `add_log_row_interval` to `row_log_interval`

### Fixed

- Fixed a bug with tensorboard logging in multi-gpu setup

## [0.4.9] - 2019-09-16

### Added

- Added the flag `log_gpu_memory` to `Trainer` to deactivate logging of GPU memory utilization
- Added SLURM resubmit functionality (port from test-tube)
- Added optional weight_save_path to trainer to remove the need for a checkpoint_callback when using cluster training
- Added option to use single gpu per node with `DistributedDataParallel`

### Changed

- Changed functionality of `validation_end` and `test_end` with multiple dataloaders to be given all of the dataloaders at once rather than in seperate calls
- Changed print_nan_grads to only print the parameter value and gradients when they contain NaN
- Changed gpu API to take integers as well (e.g. `gpus=2` instead of `gpus=[0, 1]`)
- All models now loaded on to CPU to avoid device and out of memory issues in PyTorch

### Fixed

- Fixed a bug where data types that implement `.to` but not `.cuda` would not be properly moved onto the GPU
- Fixed a bug where data would not be re-shuffled every epoch when using a `DistributedSampler`

## [0.4.8] - 2019-08-31

### Added

- Added `test_step` and `test_end` methods, used when `Trainer.test` is called
- Added `GradientAccumulationScheduler` callback which can be used to schedule changes to the number of accumulation batches
- Added option to skip the validation sanity check by setting `nb_sanity_val_steps = 0`

### Fixed

- Fixed a bug when setting `nb_sanity_val_steps = 0`

## [0.4.7] - 2019-08-24

### Changed

- Changed the default `val_check_interval` to `1.0`
- Changed defaults for `nb_val_batches`, `nb_tng_batches` and `nb_test_batches` to 0

### Fixed

- Fixed a bug where the full validation set as used despite setting `val_percent_check`
- Fixed a bug where an `Exception` was thrown when using a data set containing a single batch
- Fixed a bug where an `Exception` was thrown if no `val_dataloader` was given
- Fixed a bug where tuples were not properly transfered to the GPU
- Fixed a bug where data of a non standard type was not properly handled by the trainer
- Fixed a bug when loading data as a tuple
- Fixed a bug where `AttributeError` could be suppressed by the `Trainer`

## [0.4.6] - 2019-08-15

### Added

- Added support for data to be given as a `dict` or `list` with a single gpu
- Added support for `configure_optimizers` to return a single optimizer, two list (optimizers and schedulers), or a single list

### Fixed

- Fixed a bug where returning just an optimizer list (i.e. without schedulers) from `configure_optimizers` would throw an `Exception`

## [0.4.5] - 2019-08-13

### Added

- Added `optimizer_step` method that can be overridden to change the standard optimizer behaviour

## [0.4.4] - 2019-08-12

### Added

- Added supoort for multiple validation dataloaders
- Added support for latest test-tube logger (optimised for `torch==1.2.0`)

### Changed

- `validation_step` and `val_dataloader` are now optional
- `lr_scheduler` is now activated after epoch

### Fixed

- Fixed a bug where a warning would show when using `lr_scheduler` in `torch>1.1.0`
- Fixed a bug where an `Exception` would be thrown if using `torch.DistributedDataParallel` without using a `DistributedSampler`, this now throws a `Warning` instead

## [0.4.3] - 2019-08-10

### Fixed

- Fixed a bug where accumulate gradients would scale the loss incorrectly

## [0.4.2] - 2019-08-08

### Changed

- Changed install requirement to `torch==1.2.0`

## [0.4.1] - 2019-08-08

### Changed

- Changed install requirement to `torch==1.1.0`

## [0.4.0] - 2019-08-08

### Added

- Added 16-bit support for a single GPU
- Added support for training continuation (preserves epoch, global step etc.)

### Changed

- Changed `training_step` and `validation_step`, outputs will no longer be automatically reduced

### Removed

- Removed need for `Experiment` object in `Trainer`

### Fixed

- Fixed issues with reducing outputs from generative models (such as images and text)

## [0.3.6] - 2019-07-25

### Added

- Added a decorator to do lazy data loading internally

### Fixed

- Fixed a bug where `Experiment` object was not process safe, potentially causing logs to be overwritten

## [0.3.5] - 2019-07-25

## [0.3.4] - 2019-07-22

## [0.3.3] - 2019-07-22

## [0.3.2] - 2019-07-21

## [0.3.1] - 2019-07-21

## [0.2.x] - 2019-07-09

## [0.1.x] - 2019-06-DD

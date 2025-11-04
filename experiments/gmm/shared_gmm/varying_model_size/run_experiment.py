import argparse
from datetime import datetime
from pathlib import Path

import jax

from diffusionlab.vector_fields import VectorFieldType
from diffusionlab.samplers import DDMSampler

from diffusion_mem_gen.models.gmm import IsoHomGMMInitStrategy
from common.orchestrate import (
    aggregate_metrics,
    generate_base_config,
    generate_training_configs,
    generate_baseline_configs,
    generate_data,
    run_baseline_configs,
    run_training_configs,
)
from common.log_plot import log_and_plot_aggregated_metrics


def positive_or_default(value: int, fallback: int) -> int:
    """Return ``value`` if positive, otherwise ``fallback``."""

    return value if value > 0 else fallback


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the shared-GMM varying-model-size experiment."
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="缩小数据量与训练轮次，便于在本地快速验证端到端流程。",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="禁用 W&B 日志记录（默认开启）。",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    experiment_name = f"data_gmm_model_shared_gmm_expt_varying_model_size_{timestamp}"

    experiment_dir = Path(__file__).parent / "results" / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    num_gpus = max(1, len(jax.devices()))

    # === Configuration ===
    seed = 1
    dim = 100
    gt_num_components = 4
    gt_means_scale = dim ** 0.5
    gt_var_scale = 1.0
    num_samples_train = 100
    num_samples_val = 100
    num_samples_eval = 50
    memorization_ratio_threshold = 1 / 3
    t_min = 0.001
    t_max = 0.999
    num_times = 26
    vf_type = VectorFieldType.EPS
    num_noise_draws_per_sample = 100
    num_epochs_train = 10
    init_lr = 0.0
    peak_lr = 1e-3
    final_lr = 1e-6
    adam_b1 = 0.9
    adam_b2 = 0.999
    adam_eps = 1e-8
    gradient_clipping = 1.0e9
    ema = 0.99
    init_strategy = IsoHomGMMInitStrategy.PMEM
    init_means_scale = gt_means_scale
    init_var_scale = 1e-6
    init_means_noise_var = 0.0

    if args.quick_test:
        dim = 32
        gt_means_scale = dim ** 0.5
        num_samples_train = 32
        num_samples_val = 32
        num_samples_eval = 16
        num_times = 8
        num_noise_draws_per_sample = 16
        num_epochs_train = 2

    batch_size = num_samples_train
    num_epochs_per_log = positive_or_default(num_epochs_train // 10, 1)
    train_skip = positive_or_default(num_samples_train // 10, 1)
    eval_skip = positive_or_default(num_samples_train // 100, 1)

    if args.quick_test:
        num_components_to_train = [min(num_samples_train, 8)]
        num_components_to_evaluate = [min(num_samples_train, 8)]
    else:
        num_components_to_train = [10, 50, 100]
        num_components_to_evaluate = [10, 50, 100]

    sampler_class = DDMSampler  # For RatioOfDistancesMetric

    num_steps_train = num_epochs_train * num_samples_train // batch_size
    num_steps_warmup = positive_or_default(num_steps_train // 20, 1)
    num_steps_decay = positive_or_default(num_steps_train - num_steps_warmup, 1)

    key = jax.random.PRNGKey(seed)
    overall_data_key, overall_train_key, overall_eval_key = jax.random.split(key, 3)

    config = {
        "experiment_group_name": experiment_name + "_GROUP_TEST",
        "experiment_name": experiment_name,
        "seed": seed,
        "overall_data_key": overall_data_key,
        "overall_train_key": overall_train_key,
        "overall_eval_key": overall_eval_key,
        "dim": dim,
        "gt_num_components": gt_num_components,
        "gt_means_scale": gt_means_scale,
        "gt_var_scale": gt_var_scale,
        "num_samples_train": num_samples_train,
        "num_samples_val": num_samples_val,
        "num_samples_eval": num_samples_eval,
        "memorization_ratio_threshold": memorization_ratio_threshold,
        "t_min": t_min,
        "t_max": t_max,
        "num_times": num_times,
        "vf_type": vf_type,
        "num_noise_draws_per_sample": num_noise_draws_per_sample,
        "num_epochs_train": num_epochs_train,
        "num_epochs_per_log": num_epochs_per_log,
        "num_steps_train": num_steps_train,
        "num_steps_warmup": num_steps_warmup,
        "num_steps_decay": num_steps_decay,
        "batch_size": batch_size,
        "init_lr": init_lr,
        "peak_lr": peak_lr,
        "final_lr": final_lr,
        "adam_b1": adam_b1,
        "adam_b2": adam_b2,
        "adam_eps": adam_eps,
        "gradient_clipping": gradient_clipping,
        "ema": ema,
        "init_strategy": init_strategy,
        "init_means_scale": init_means_scale,
        "init_var_scale": init_var_scale,
        "init_means_noise_var": init_means_noise_var,
        "train_skip": train_skip,
        "eval_skip": eval_skip,
        "num_components_to_train": num_components_to_train,
        "num_components_to_evaluate": num_components_to_evaluate,
        "sampler_class": sampler_class,
    }

    use_wandb = not args.disable_wandb

    base_config_file = generate_base_config(config, experiment_dir)
    data_file = generate_data(base_config_file, experiment_dir)
    training_configs_files = generate_training_configs(base_config_file, experiment_dir)
    baseline_configs_files = generate_baseline_configs(base_config_file, experiment_dir)
    training_results_files = run_training_configs(
        training_configs_files,
        data_file,
        experiment_dir,
        max_workers=num_gpus,
        num_gpus=num_gpus,
        use_wandb=use_wandb,
    )
    baseline_results_files = run_baseline_configs(
        baseline_configs_files,
        data_file,
        experiment_dir,
        max_workers=num_gpus,
        num_gpus=num_gpus,
        use_wandb=False,
    )
    metrics_file = aggregate_metrics(
        training_results_files, baseline_results_files, experiment_dir
    )
    log_and_plot_aggregated_metrics(
        base_config_file, data_file, metrics_file, experiment_dir, use_wandb=use_wandb
    )


if __name__ == "__main__":
    main()

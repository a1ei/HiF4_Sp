#!/usr/bin/env python3
"""使用 vLLM 部署模型并在 lighteval 上做评估。

本脚本直接从 ``3rdparty/lighteval/src`` 导入 lighteval，仓库对 lighteval 的本地
改动已经随 ``3rdparty/lighteval`` 内置源码目录提交，协作者只需：

1. 安装依赖：``bash install.sh``（在 ``hif4`` 环境下，源码编译本仓库自带的
   vLLM，并安装本仓库自带的 lighteval）。
2. 运行本脚本即可。

CLI 与 ``Half-Experts-Candoall/main_backup_3rdparty_lighteval.py`` 保持一致，额外
暴露了 ``--enforce_eager`` / ``--cpu_offload_gb``（依赖本仓库对 lighteval 的
改动，见 ``3rdparty/lighteval/src/lighteval/models/vllm/vllm_model.py``）。
"""
import argparse
import json
import os
import sys
from pathlib import Path

# 优先使用仓库内的 lighteval 源码（附带本仓库的修改）；使用 resolve() 是因为
# vLLM 等库用 multiprocessing spawn 会再次执行本文件，需稳定得到仓库根目录。
REPO_ROOT = str(Path(__file__).resolve().parent)
LIGHTEVAL_SRC = os.path.join(REPO_ROOT, "3rdparty", "lighteval", "src")
if not os.path.isdir(LIGHTEVAL_SRC):
    raise RuntimeError(
        f"未找到 {LIGHTEVAL_SRC}：请确认 3rdparty/lighteval 已随仓库完整 clone。"
    )
if LIGHTEVAL_SRC not in sys.path:
    sys.path.insert(0, LIGHTEVAL_SRC)
# 让自定义任务能通过 `from tasks import ...` 被 lighteval 内部的 spec loader 找到
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def get_short_model_name(model_path: str) -> str:
    """从模型路径或 HF id 得到简短名称，仅用最后一级（如 Qwen3-30B-A3B-Instruct-2507）。"""
    return model_path.rstrip("/").split("/")[-1]


def _get_moe_num_experts_per_tok(config: dict) -> int | None:
    """从 config 中读取 num_experts_per_tok，兼容顶层（Qwen3）与 text_config 下（Qwen3.5）两种格式。"""
    if "num_experts_per_tok" in config:
        return config["num_experts_per_tok"]
    text_cfg = config.get("text_config")
    if isinstance(text_cfg, dict) and "num_experts_per_tok" in text_cfg:
        return text_cfg["num_experts_per_tok"]
    return None


def _set_moe_num_experts_per_tok(config: dict, num_experts_per_tok: int) -> None:
    """向 config 写入 num_experts_per_tok，兼容顶层（Qwen3）与 text_config 下（Qwen3.5）两种格式。"""
    text_cfg = config.get("text_config")
    if isinstance(text_cfg, dict) and "num_experts_per_tok" in text_cfg:
        text_cfg["num_experts_per_tok"] = num_experts_per_tok
    else:
        config["num_experts_per_tok"] = num_experts_per_tok


def prepare_model_dir_with_num_experts_per_tok(model_path: str, num_experts_per_tok: int) -> str:
    """为 MoE 模型创建一份仅修改 ``num_experts_per_tok`` 的「视图」目录（其余文件 symlink）。

    返回该目录路径，供 vLLM 加载。仅当 ``model_path`` 为本地目录且存在
    ``config.json`` 时有效。兼容 Qwen3（顶层 ``num_experts_per_tok``）与
    Qwen3.5（``text_config.num_experts_per_tok``）两种 config 格式。
    """
    model_path = os.path.abspath(model_path)
    if not os.path.isdir(model_path):
        raise ValueError(f"num_experts_per_tok 仅支持本地模型目录，当前 model_path 不是目录: {model_path}")
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        raise ValueError(f"未找到 config.json: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if _get_moe_num_experts_per_tok(config) is None:
        raise ValueError("config.json 中无 num_experts_per_tok（顶层或 text_config 下），当前可能不是 MoE 模型")
    _set_moe_num_experts_per_tok(config, num_experts_per_tok)
    override_root = os.path.join(REPO_ROOT, ".moe_override")
    short_name = get_short_model_name(model_path)
    dest_dir = os.path.join(override_root, f"{short_name}_k{num_experts_per_tok}")
    os.makedirs(dest_dir, exist_ok=True)
    for name in os.listdir(model_path):
        src = os.path.join(model_path, name)
        dst = os.path.join(dest_dir, name)
        if name == "config.json":
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        elif not os.path.exists(dst):
            os.symlink(src, dst)
    return dest_dir


def maybe_register_hif4_flatquant_model(model_path: str) -> bool:
    model_path = os.path.abspath(model_path)
    if not os.path.isdir(model_path):
        return False
    flatquant_config = os.path.join(model_path, "hif4_flatquant_config.json")
    if not os.path.isfile(flatquant_config):
        return False

    hif4_root = os.path.join(REPO_ROOT, "HiFloat4")
    if hif4_root not in sys.path:
        sys.path.insert(0, hif4_root)
    from hif4flatquant.vllm_custom import register_hif4_flatquant_models

    register_hif4_flatquant_models()
    print(f"检测到 HiF4 FlatQuant 配置，已注册 vLLM 自定义模型: {flatquant_config}")
    return True


class CustomEvaluationTracker:
    """自定义 Tracker：结果与 details 保存在 ``output_dir/<short_model_name>/{results,details}`` 下，且 details 存为 JSON。"""

    def __init__(self, output_dir: str, short_model_name: str, save_details: bool = True, **kwargs):
        from dataclasses import asdict
        from datasets import Dataset
        from lighteval.logging.evaluation_tracker import EvaluationTracker, EnhancedJSONEncoder

        self._asdict = asdict
        self._Dataset = Dataset
        self._EnhancedJSONEncoder = EnhancedJSONEncoder
        self._short_model_name = short_model_name
        self._tracker = EvaluationTracker(output_dir=output_dir, save_details=save_details, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tracker, name)

    def save(self):
        """覆盖 save：使用 ``output_dir/<short_model_name>/{results,details}``，details 写 JSON。"""
        from datetime import datetime

        date_id = datetime.now().isoformat().replace(":", "-")
        results_dict = self._tracker.results
        details_datasets = {}
        for task_name, task_details in self._tracker.details_logger.details.items():
            dataset = self._Dataset.from_list([self._asdict(d) for d in task_details])
            col = [c for c in dataset.column_names if c != "id"] or dataset.column_names
            dataset = dataset.select_columns(sorted(col))
            details_datasets[task_name] = dataset
        self.save_results(date_id, results_dict)
        if self._tracker.should_save_details:
            self.save_details(date_id, details_datasets)
        if self._tracker.should_push_to_hub:
            self._tracker.push_to_hub(date_id=date_id, details=details_datasets, results_dict=results_dict)
        if getattr(self._tracker, "use_wandb", False):
            self._tracker.push_to_wandb(results_dict=results_dict, details_datasets=details_datasets)
        if getattr(self._tracker, "should_push_results_to_tensorboard", False):
            self._tracker.push_to_tensorboard(
                results=self._tracker.metrics_logger.metric_aggregated,
                details=self._tracker.details_logger.compiled_details,
            )

    def save_results(self, date_id: str, results_dict: dict):
        output_dir_results = Path(self._tracker.output_dir) / self._short_model_name / "results"
        self._tracker.fs.mkdirs(str(output_dir_results), exist_ok=True)
        output_results_file = output_dir_results / f"results_{date_id}.json"
        with self._tracker.fs.open(str(output_results_file), "w") as f:
            f.write(json.dumps(results_dict, cls=self._EnhancedJSONEncoder, indent=2, ensure_ascii=False))

    def _get_gold_from_doc(self, doc: dict):
        """从 doc 的 ``choices + gold_index`` 解析出参考答案，与 lighteval ``Doc.get_golds()`` 语义一致。"""
        choices = doc.get("choices")
        gold_index = doc.get("gold_index")
        if choices is None or gold_index is None:
            return None
        gold_indices = [gold_index] if isinstance(gold_index, int) else gold_index
        golds = []
        for ix in gold_indices:
            if ix < 0 or ix >= len(choices):
                continue
            c = choices[ix]
            if c is None:
                continue
            if isinstance(c, list):
                golds.extend(c)
            else:
                golds.append(c)
        return golds if golds else None

    def _filter_detail_record(self, record: dict) -> dict:
        """每条 detail 保留：``doc.id``、``doc.specific``、``doc.gold``、``metric``、``model_response``，保证各类任务都能看到 gold。"""
        doc = record.get("doc") or {}
        model_response = record.get("model_response") or {}
        gold = self._get_gold_from_doc(doc)
        out = {
            "doc": {"id": doc.get("id"), "specific": doc.get("specific")},
            "metric": record.get("metric"),
            "model_response": {
                "input": model_response.get("input"),
                "text_post_processed": model_response.get("text_post_processed"),
            },
        }
        if gold is not None:
            out["gold"] = gold
        return out

    def save_details(self, date_id: str, details_datasets: dict):
        output_dir_details_sub_folder = Path(self._tracker.output_dir) / self._short_model_name / "details" / date_id
        self._tracker.fs.mkdirs(str(output_dir_details_sub_folder), exist_ok=True)
        for task_name, dataset in details_datasets.items():
            output_file = output_dir_details_sub_folder / f"details_{task_name}_{date_id}.json"
            records = dataset.to_list() if hasattr(dataset, "to_list") else [dataset[i] for i in range(len(dataset))]
            filtered = [self._filter_detail_record(r) for r in records]
            with self._tracker.fs.open(str(output_file), "w") as f:
                f.write(json.dumps(filtered, indent=2, ensure_ascii=False, default=str))


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM + lighteval 评估脚本")
    parser.add_argument(
        "--datasets",
        type=str,
        default="gsm8k",
        help="评估数据集，逗号分隔，如 a,b,c。默认: gsm8k",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="vLLM 张量并行 (TP)：单层内参数切分到的 GPU 数。默认: 当前可见 GPU 数",
    )
    parser.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=None,
        help="vLLM 流水线并行 (PP)：按层切分到多卡，降低单卡显存；通常与 --tensor_parallel_size 联用，使 TP×PP 等于所用 GPU 总数。不设则为 1",
    )
    parser.add_argument(
        "--data_parallel_size",
        type=int,
        default=None,
        help="vLLM 数据并行副本数（多份完整模型，lighteval 会启用 Ray）。主要用于吞吐，一般不减单卡显存。不设则为 1",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32768,
        help="生成最大 token 数。默认: 32768",
    )
    parser.add_argument(
        "--max_model_length",
        "--max_model_len",
        type=int,
        default=32768,
        dest="max_model_length",
        help="模型最大序列长度（与 vLLM max_model_len 一致）。--max_model_len 为同义简写。默认: 32768",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="每个任务最多评估样本数，不设则全量。默认: 不设置",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度。默认: 0.7",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="top_p 采样。默认: 0.8",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="top_k 采样。默认: 20",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="vLLM 采样随机种子。默认: 1234",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="结果与 details 保存目录。默认: ./results",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="模型路径或 HuggingFace 模型 id。默认: Qwen/Qwen3-30B-A3B-Instruct-2507",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.95,
        help="vLLM GPU 显存利用率。默认: 0.95",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="vLLM max_num_seqs：单轮迭代中最大并行序列数（与 EngineArgs.max_num_seqs 一致）。不设则使用 lighteval/vLLM 默认值",
    )
    parser.add_argument(
        "--custom_tasks",
        type=str,
        default=None,
        help="自定义任务 Python 文件路径。默认: 仓库内 tasks/custom_tasks.py（聚合 aime/triviaqa/simpleqa/hellaswag/if_pass_at_n 全部任务）",
    )
    parser.add_argument(
        "--num_experts_per_tok",
        type=int,
        default=None,
        help="MoE 模型每 token 激活的专家数，覆盖 config.json。仅支持本地模型目录。用于 pruning 实验（如 k,k-1,...,1）",
    )
    parser.add_argument(
        "--load_multilingual_tasks",
        action="store_true",
        help="加载 lighteval 多语言任务注册表（含 ceval/cmmlu/agieval 等）。评 C-Eval 等时必须加此选项，与官方 CLI 的 --load-tasks-multilingual 一致",
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="传给 vLLM：enforce_eager=True（关闭 CUDAGraph 等，便于排错但更慢）。默认 False，与 vLLM 一致。"
             "依赖本仓库对 lighteval 的本地修改（VLLMModelConfig.enforce_eager 字段）。",
    )
    parser.add_argument(
        "--cpu_offload_gb",
        type=float,
        default=0,
        help="每张 GPU 将多少 GiB 的模型权重卸载到 CPU 内存。默认 0（不卸载）。"
             "当 GPU 显存不够装完整权重时使用，例如 4×A800-80G 跑 235B 模型可设 --cpu_offload_gb 40。"
             "注意：会显著降低推理速度。依赖本仓库对 lighteval 的本地修改（VLLMModelConfig.cpu_offload_gb 字段）。",
    )
    parser.add_argument(
        "--fake_act_quant",
        choices=["none", "hif4", "nvfp4", "hif4-1"],
        default="none",
        help="vLLM 普通 dense linear 输入激活 fake quant 格式。默认 none。",
    )
    parser.add_argument(
        "--fp32_weights_bf16_activations",
        action="store_true",
        help="Linear 权重保持 FP32；层间激活保持 BF16，GEMM 前临时转 FP32，输出转回 BF16。",
    )
    parser.add_argument(
        "--fp32_weights_fp32_activations",
        action="store_true",
        help="整个模型使用 FP32 权重和 FP32 激活运行。",
    )
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="强制使用 tokenizer 的 chat template 构造评测 prompt。",
    )
    parser.add_argument(
        "--kv_quant_format",
        choices=["none", "nvfp4", "hif4", "hif4-1"],
        default="none",
        help="KV cache 伪量化格式。默认: none",
    )
    parser.add_argument(
        "--kv_quant_chunk_size",
        type=int,
        default=64,
        help="NVFP4 KV 伪量化的 non-sink token chunk 大小；hif4 会忽略该参数。默认: 64",
    )
    parser.add_argument(
        "--kv_quant_sink_size",
        type=int,
        default=4,
        help="前多少个 token 的 KV cache 保持原精度。默认: 4",
    )
    parser.add_argument(
        "--kv_quant_target",
        choices=["kv", "k", "v"],
        default="kv",
        help="KV 伪量化目标。默认: kv",
    )
    parser.add_argument(
        "--kv_quant_query",
        choices=["none", "enabled"],
        default="none",
        help="是否额外量化 Q。默认 none；enabled 时在 RoPE 后、QK 点积前按 kv_quant_format 做 Q 伪量化。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed < 0:
        raise ValueError("--seed 须为非负整数")
    if (
        args.fp32_weights_bf16_activations
        and args.fp32_weights_fp32_activations
    ):
        raise ValueError("两种 FP32 权重模式不能同时启用")
    if (
        args.fp32_weights_bf16_activations
        or args.fp32_weights_fp32_activations
    ) and args.fake_act_quant not in ("none", "nvfp4"):
        raise ValueError(
            "FP32 权重模式只支持 --fake_act_quant none 或 nvfp4。"
        )
    if args.kv_quant_format == "nvfp4" and args.kv_quant_chunk_size < 1:
        raise ValueError("--kv_quant_chunk_size 须为正整数")
    if args.kv_quant_sink_size < 0:
        raise ValueError("--kv_quant_sink_size 须为非负整数")

    # 若指定 num_experts_per_tok，为本地 MoE 模型准备覆盖目录并替换 model_path
    if args.num_experts_per_tok is not None:
        args.model_path = prepare_model_dir_with_num_experts_per_tok(
            args.model_path, args.num_experts_per_tok
        )
        print(f"MoE 已覆盖 num_experts_per_tok={args.num_experts_per_tok}，使用目录: {args.model_path}")

    is_hif4_flatquant = maybe_register_hif4_flatquant_model(args.model_path)
    if is_hif4_flatquant and args.tensor_parallel_size not in (None, 1):
        raise ValueError("HiF4 FlatQuant vLLM 自定义推理当前只支持 --tensor_parallel_size 1。")

    nvf4_activation_scales_path = None
    if args.fake_act_quant == "nvfp4":
        model_path = os.path.abspath(args.model_path)
        if not os.path.isdir(model_path):
            raise ValueError(f"--fake_act_quant nvfp4 只支持本地 model_path 目录: {args.model_path}")
        nvf4_activation_scales_path = os.path.join(
            model_path,
            "nvfp4_activation_scales.safetensors",
        )
        if not os.path.isfile(nvf4_activation_scales_path):
            raise FileNotFoundError(
                "--fake_act_quant nvfp4 需要转换时保存的 activation scale 文件: "
                f"{nvf4_activation_scales_path}"
            )

    # tensor_parallel_size 未设置时取当前可见 GPU 数
    tensor_parallel_size = args.tensor_parallel_size
    if tensor_parallel_size is None:
        try:
            import torch

            tensor_parallel_size = torch.cuda.device_count()
        except Exception:
            tensor_parallel_size = 1
        if tensor_parallel_size <= 0:
            tensor_parallel_size = 1

    from lighteval.models.model_input import GenerationParameters
    from lighteval.models.vllm.vllm_model import VLLMModelConfig
    from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

    short_model_name = get_short_model_name(args.model_path)
    evaluation_tracker = CustomEvaluationTracker(
        output_dir=args.output_dir,
        short_model_name=short_model_name,
        save_details=True,
    )
    if args.custom_tasks == "":
        custom_tasks_path = None
    elif args.custom_tasks is None:
        custom_tasks_path = os.path.join(REPO_ROOT, "tasks", "custom_tasks.py")
    else:
        custom_tasks_path = args.custom_tasks
    kwargs = dict(
        launcher_type=ParallelismManager.VLLM,
        max_samples=args.max_samples,
        load_tasks_multilingual=args.load_multilingual_tasks,
    )
    if custom_tasks_path and os.path.isfile(custom_tasks_path):
        kwargs["custom_tasks_directory"] = os.path.abspath(custom_tasks_path)
    pipeline_params = PipelineParameters(**kwargs)

    kv_quant_enabled = args.kv_quant_format != "none"
    if args.fp32_weights_fp32_activations:
        model_dtype = "float32"
    elif args.fp32_weights_bf16_activations:
        model_dtype = "bfloat16"
    else:
        model_dtype = "auto"

    vllm_model_kwargs = dict(
        model_name=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_length=args.max_model_length,
        dtype=model_dtype,
        enforce_eager=args.enforce_eager,
        cpu_offload_gb=args.cpu_offload_gb,
        seed=args.seed,
        generation_parameters=GenerationParameters(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
        ),
    )
    if args.use_chat_template:
        vllm_model_kwargs["override_chat_template"] = True
    additional_config = {}
    if args.fp32_weights_bf16_activations:
        additional_config["fp32_weights_bf16_activations"] = True
    if args.fake_act_quant != "none":
        additional_config["fake_act_quant"] = args.fake_act_quant
    if args.fake_act_quant == "nvfp4":
        additional_config.update(
            {
                "nvf4_activation_scales_path": nvf4_activation_scales_path,
            }
        )
    if kv_quant_enabled:
        additional_config.update(
            {
                "kv_quant_format": args.kv_quant_format,
                "kv_quant_chunk_size": args.kv_quant_chunk_size,
                "kv_quant_sink_size": args.kv_quant_sink_size,
                "kv_quant_target": args.kv_quant_target,
                "kv_quant_query": args.kv_quant_query,
            }
        )
    if additional_config:
        vllm_model_kwargs["additional_config"] = additional_config
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch_size 须为正整数")
        vllm_model_kwargs["max_num_seqs"] = args.batch_size
    if args.pipeline_parallel_size is not None:
        if args.pipeline_parallel_size < 1:
            raise ValueError("--pipeline_parallel_size 须为 >= 1 的整数")
        vllm_model_kwargs["pipeline_parallel_size"] = args.pipeline_parallel_size
    if args.data_parallel_size is not None:
        if args.data_parallel_size < 1:
            raise ValueError("--data_parallel_size 须为 >= 1 的整数")
        vllm_model_kwargs["data_parallel_size"] = args.data_parallel_size
    model_config = VLLMModelConfig(**vllm_model_kwargs)

    pipeline = Pipeline(
        tasks=args.datasets,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    par = f"TP={tensor_parallel_size}"
    if args.pipeline_parallel_size is not None:
        par += f", PP={args.pipeline_parallel_size}"
    if args.data_parallel_size is not None:
        par += f", DP={args.data_parallel_size}"
    print(
        f"使用 vLLM 后端评估: datasets={args.datasets}, model={args.model_path} "
        f"(保存名: {short_model_name}), 参数并行: {par}"
    )
    pipeline.evaluate()
    pipeline.show_results()
    results = pipeline.get_results()
    details = pipeline.get_details()

    # 保存前用简短模型名，使 results 里的 config 也一致
    evaluation_tracker.general_config_logger.model_name = short_model_name
    pipeline.save_and_push_results()

    print(f"评估完成。结果与 details 已保存至: {args.output_dir}/{short_model_name}/")
    return results, details


if __name__ == "__main__":
    main()

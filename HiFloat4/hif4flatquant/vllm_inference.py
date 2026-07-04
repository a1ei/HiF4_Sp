import argparse
import pathlib
import sys

HIF4_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(HIF4_ROOT) not in sys.path:
    sys.path.insert(0, str(HIF4_ROOT))

from hif4flatquant.vllm_custom import register_hif4_flatquant_models


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tensor_parallel_size != 1:
        raise ValueError("HiF4 FlatQuant vLLM inference currently requires tensor_parallel_size=1.")
    register_hif4_flatquant_models()
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    outputs = llm.generate([args.prompt], SamplingParams(max_tokens=args.max_tokens))
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()

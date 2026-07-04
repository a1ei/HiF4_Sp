from vllm import ModelRegistry


def register_hif4_flatquant_models():
    ModelRegistry.register_model(
        "Qwen3_5Hif4FlatQuantForCausalLM",
        "hif4flatquant.vllm_custom.qwen3_5:Qwen3_5Hif4FlatQuantForCausalLM",
    )

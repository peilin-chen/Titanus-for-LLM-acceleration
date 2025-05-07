# Titanus: Enabling KV Cache Pruning and Quantization On-the-Fly for LLM Acceleration
This repository contains the code asscoiated with "Titanus: Enabling KV Cache Pruning and Quantization On-the-Fly for LLM Acceleration", accepted to GLSVLSI2025. It contains the cycle-accurate simulator for Titanus architecture.

## Introduction
Large language models (LLMs) have gained great success in various domains. Existing systems cache Key and Value within the attention block to avoid redundant computations. However, the size of key-value cache (KV cache) is unpredictable and can even be tens of times larger than the weights in the long context length scenario. In this work, we propose Titanus, a software-hardware co-design to efficiently compress the KV cache on-the-fly. We first propose the cascade pruning-quantization (CPQ) method to reduce the KV cache movement. The hierarchical quantization extension strategy is introduced to tackle the non-independent per-channel quantization issue. To further reduce KV cache movement, we transfer only the non-zero KV cache between the accelerator and off-chip memory. Moreover, we customize a two-stage design space exploration framework for the CPQ method. A novel pipeline and parallelism dataflow is designed to reduce the first token generation time. Experiments show that Titanus achieves 159.9× (49.6×) and 34.8× (29.2×) energy efficiency (throughput) compared to Nvidia A100 GPU and FlightLLM respectively. 

## Run Simulator
Use the following command to get started:
```shell
python ./run_simulator.py --model_dict_path ./model_dicts/model-opt-125m.json --config_path ./config/config_opt125m_32_32_TT.yaml --constants_path ./constant/constants.yaml
```
The output will look like this:
```shell
bash-4.4$python ./run_simulator.py --model_dict_path ./model_dicts/model-opt-125m.json --config_path ./config/config_opt125m_32_32_TT.yaml --constants_path ./constant/constants.yaml
Accelerator area (one core):  83.3200 mm² (Total area: core_area×core_num -> 83.3200 × 12)
Accelerator frequency: 200 MHz
Pruning enable: True
Quantization enable: True
------------------Core-level Component-wise Area Breakdown------------------
- Buffer Area:  1.1952 mm² (1.435%) -> (global buffer: 4000KB, sz buffer: 64.0KB)
- DCIM Area:  80.5770 mm² (96.708%)
- Computing Engine Area:  0.9904 mm² (1.189%)
- Pruning Unit Area:  0.0014 mm² (0.002%)
- Quantization Unit Area:  0.1901 mm² (0.228%)
- Dequantization Unit Area:  0.0042 mm² (0.005%)
- Softmax Area:  0.1228 mm² (0.147%)
- LayerNorm Area:  0.2350 mm² (0.282%)
- ReLU Area:  0.0038 mm² (0.005%)
------------------Core-level Component-wise Power Breakdown------------------
- Total Power:  31950.1496 mW
- Buffer Power:  893.6821 mW (2.797%)
- DCIM Power:  30880.4604 mW (96.652%)
- Computing Engine Power:  100.8504 mW (0.316%)
- Pruning Unit Power:  0.2147 mW (0.001%)
- Quantization Unit Power:  51.7940 mW (0.162%)
- Dequantization Unit Power:  1.5040 mW (0.005%)
- Softmax Power:  9.6518 mW (0.030%)
- LayerNorm Power:  11.6430 mW (0.036%)
- ReLU Power:  0.3491 mW (0.001%)
------------------LLM configurations------------------
Prefill stage sequence length: 32
Decode stage sequence length: 32
Layer number: 12
Attention head number: 12
Hidden size: 768
FFN dimension: 3072
Converting model to hardware operations: 100%|███████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 918.15it/s]
------------------Energy------------------
Compute energy: 815760.829216 nJ
Memory energy: 39826925.416744 nJ
- global buffer energy: 39826890.171954 nJ
---- weight access energy: 35254615.903444 nJ (88.520%)
---- KV cache access energy: 4449862.407734 nJ (11.173%)
---- other access energy: 122411.860776 nJ (0.307%)
- sz buffer energy: 35.244790 nJ
Total energy: 40642686.245959 nJ
------------------Performance------------------
Energy Efficiency (Token/J): 11878.09 Token/J
Energy Efficiency (uJ/Token): 84.19 uJ/Token
Throughput (original): 10375.33 Token/s
Latency (original): 6.168480 ms
Throughput (intra-pipeline): 14717.52 Token/s (speedup compared to original: 1.42)
Latency (intra-pipeline): 4.348560 ms (reduction compared to original: 29.504%)
Throughput (intra-pipeline+inter-parallelism): 18801.47 Token/s (speedup compared to intra-pipeline: 1.28)
Latency (intra-pipeline+inter-parallelism): 3.403990 ms (reduction compared to intra-pipeline: 21.721%)
Finish simulation!
```

## Citation
```
To be appear!
```

## Reference Repositories
acceltran: https://github.com/jha-lab/acceltran

cacti: https://github.com/HewlettPackard/cacti

NVmain: https://github.com/SEAL-UCSB/NVmain


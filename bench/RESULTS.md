# Benchmark results

Hardware: Intel(R) Core(TM) Ultra 7 155H; devices: CPU = Intel(R) Core(TM) Ultra 7 155H, GPU = Intel(R) Arc(TM) Graphics (iGPU), NPU = Intel(R) AI Boost; OpenVINO 2026.3.1-22476-759c5a6ab8c-releases/2026/3.


## perception

| ms_per_look | resolution |
|---|---|
| 154.6 | 960x720 colour + depth |

## act

| backend | precision | device | ms_per_call | calls_per_s | actions_per_s | speedup_vs_pytorch |
|---|---|---|---|---|---|---|
| PyTorch | FP32 | CPU | 2.065 | 484.3 | 9685 | 1.0 |
| OpenVINO | FP32 | CPU | 0.831 | 1203.4 | 24067 | 2.48 |
| OpenVINO | FP32 | GPU | 1.489 | 671.6 | 13432 | 1.39 |
| OpenVINO | FP32 | NPU | 0.903 | 1107.4 | 22148 | 2.29 |
| OpenVINO | FP16 | CPU | 0.806 | 1240.7 | 24814 | 2.56 |
| OpenVINO | FP16 | GPU | 1.522 | 657.0 | 13141 | 1.36 |
| OpenVINO | FP16 | NPU | 1.06 | 943.4 | 18868 | 1.95 |
| OpenVINO | INT8 | CPU | 0.724 | 1381.2 | 27624 | 2.85 |
| OpenVINO | INT8 | GPU | 1.419 | 704.7 | 14094 | 1.46 |
| OpenVINO | INT8 | NPU | 1.099 | 909.9 | 18198 | 1.88 |
| OpenVINO | INT8W | CPU | 0.743 | 1345.9 | 26918 | 2.78 |
| OpenVINO | INT8W | GPU | 1.929 | 518.4 | 10368 | 1.07 |
| OpenVINO | INT8W | NPU | 1.036 | 965.3 | 19305 | 1.99 |

## act_task

| policy | within_1_5cm | within_2_5cm |
|---|---|---|
| PyTorch FP32 | 9/10 | 9/10 |
| OpenVINO FP32 (CPU) | 9/10 | 9/10 |
| OpenVINO FP16 (CPU) | 9/10 | 9/10 |
| OpenVINO INT8 (CPU) | 9/10 | 9/10 |
| OpenVINO INT8W (CPU) | 10/10 | 10/10 |
| OpenVINO FP16 (GPU) | 10/10 | 10/10 |
| OpenVINO INT8 (GPU) | 8/10 | 10/10 |
| OpenVINO FP16 (NPU) | 10/10 | 10/10 |
| OpenVINO INT8 (NPU) | 8/10 | 9/10 |

## vlm

| device | image_width | load_s | s_per_command | first_token_ms | tokens_per_s | correct |
|---|---|---|---|---|---|---|
| CPU | 320 | 4.1 | 3.0 | 2065 | 12.5 | 5/5 |
| CPU | 480 | 4.1 | 3.6 | 2707 | 12.7 | 5/5 |
| CPU | 960 | 4.1 | 10.9 | 9910 | 11.4 | 5/5 |
| GPU | 320 | 12.8 | 1.1 | 564 | 21.1 | 5/5 |
| GPU | 480 | 12.8 | 1.3 | 696 | 20.7 | 5/5 |
| GPU | 960 | 12.8 | 2.8 | 2193 | 19.5 | 5/5 |

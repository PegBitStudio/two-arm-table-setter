# Benchmark results

Hardware: 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz; devices: CPU = 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz, GPU = Intel(R) Iris(R) Xe Graphics (iGPU); OpenVINO 2026.3.1-22476-759c5a6ab8c-releases/2026/3.


## vlm

| device | image_width | load_s | s_per_command | first_token_ms | tokens_per_s | correct |
|---|---|---|---|---|---|---|
| CPU | 320 | 4.8 | 3.8 | 2273 | 7.9 | 5/5 |
| CPU | 480 | 4.8 | 4.3 | 2712 | 7.2 | 5/5 |
| CPU | 960 | 4.8 | 14.9 | 13159 | 6.7 | 5/5 |
| GPU | 320 | 17.2 | 4.0 | 1385 | 4.5 | 5/5 |
| GPU | 480 | 17.2 | 4.2 | 1612 | 4.5 | 5/5 |
| GPU | 960 | 17.2 | 7.5 | 4831 | 4.5 | 5/5 |

## perception

| ms_per_look | resolution |
|---|---|
| 889.3 | 960x720 colour + depth |

## act

| backend | precision | device | ms_per_call | speedup_vs_pytorch | calls_per_s | actions_per_s |
|---|---|---|---|---|---|---|
| PyTorch | FP32 | CPU | 4.339 | 1.0 | 230.5 | 4609 |
| OpenVINO | FP32 | CPU | 1.727 | 2.51 | 579.0 | 11581 |
| OpenVINO | FP32 | GPU | 12.378 | 0.35 | 80.8 | 1616 |
| OpenVINO | FP16 | CPU | 1.877 | 2.31 | 532.8 | 10655 |
| OpenVINO | FP16 | GPU | 14.611 | 0.3 | 68.4 | 1369 |
| OpenVINO | INT8 | CPU | 3.684 | 1.18 | 271.4 | 5429 |
| OpenVINO | INT8 | GPU | 12.861 | 0.34 | 77.8 | 1555 |
| OpenVINO | INT8W | CPU | 3.53 | 1.23 | 283.3 | 5666 |
| OpenVINO | INT8W | GPU | 23.488 | 0.18 | 42.6 | 851 |

## act_task

| policy | within_1_5cm | within_2_5cm |
|---|---|---|
| PyTorch FP32 | 9/10 | 9/10 |
| OpenVINO FP32 (CPU) | 9/10 | 9/10 |
| OpenVINO FP16 (CPU) | 9/10 | 9/10 |
| OpenVINO INT8 (CPU) | 9/10 | 10/10 |
| OpenVINO INT8W (CPU) | 10/10 | 10/10 |

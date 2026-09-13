# Benchmark results

Hardware: 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz; devices: CPU = 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz, GPU = Intel(R) Iris(R) Xe Graphics (iGPU); OpenVINO 2026.3.1-22476-759c5a6ab8c-releases/2026/3.


## vlm

| device | image_width | load_s | s_per_command | first_token_ms | tokens_per_s | correct |
|---|---|---|---|---|---|---|
| GPU | 480 | 37.6 | 5.2 | 1795 | 3.4 | 5/5 |

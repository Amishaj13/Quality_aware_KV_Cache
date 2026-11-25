# KV-cache methods vs the full cache

Quality: pass@1 (% of executable tasks solved) for `custom` and `position`; mean LongCodeU score × 100 for `longcodeu`. Retention = method ÷ full × 100. Memory = KV-cache bytes actually stored at the end of generation, averaged over the compared samples.

Small samples (30 custom tasks, 80 LongCodeU, 150 position prompts): differences of a few points are within noise.

| workload   | method     |   n_compared |   quality_full |   quality_method |   quality_retention_pct |   kv_full_mib |   kv_method_mib |   memory_reduction_pct |   compression_ratio |   peak_gpu_gib_full |   peak_gpu_gib_method |   tpot_ms_full |   tpot_ms_method |   mean_input_tokens |   snapkv_compressed_share_pct |
|:-----------|:-----------|-------------:|---------------:|-----------------:|------------------------:|--------------:|----------------:|-----------------------:|--------------------:|--------------------:|----------------------:|---------------:|-----------------:|--------------------:|------------------------------:|
| custom     | kivi4      |           30 |           61.1 |             66.7 |                   109.1 |          21.3 |             9.2 |                   56.8 |                2.32 |               14.24 |                 14.23 |           35.1 |             37   |                 239 |                           nan |
| custom     | kivi2      |           30 |           61.1 |             61.1 |                   100   |          21.3 |             7.1 |                   66.8 |                3.01 |               14.24 |                 14.23 |           35.1 |             37.8 |                 239 |                           nan |
| custom     | snapkv1024 |           30 |           61.1 |             61.1 |                   100   |          21.3 |            21.3 |                    0   |                1    |               14.24 |                 14.24 |           35.1 |             35.1 |                 239 |                             0 |
| custom     | snapkv2048 |           30 |           61.1 |             61.1 |                   100   |          21.3 |            21.3 |                    0   |                1    |               14.24 |                 14.24 |           35.1 |             35   |                 239 |                             0 |
| longcodeu  | kivi4      |           80 |           51.7 |             50.9 |                    98.5 |         258.8 |            83.1 |                   67.9 |                3.12 |               15.03 |                 14.87 |           36.5 |             42.1 |                4450 |                           nan |
| longcodeu  | kivi2      |           80 |           51.7 |             49.6 |                    96   |         258.8 |            51.1 |                   80.2 |                5.06 |               15.03 |                 14.83 |           36.5 |             43   |                4450 |                           nan |
| longcodeu  | snapkv1024 |           80 |           51.7 |             20.2 |                    39.1 |         258.8 |            70.6 |                   72.7 |                3.66 |               15.03 |                 14.84 |           36.5 |             35.3 |                4450 |                           100 |
| longcodeu  | snapkv2048 |           80 |           51.7 |             39.7 |                    76.7 |         258.8 |           126.7 |                   51.1 |                2.04 |               15.03 |                 14.91 |           36.5 |             35.6 |                4450 |                           100 |
| position   | kivi4      |          150 |           61.1 |             61.1 |                   100   |         425.8 |           135.9 |                   68.1 |                3.13 |               15.62 |                 15.36 |           37.7 |             47.2 |                7631 |                           nan |
| position   | kivi2      |          150 |           61.1 |             58.9 |                    96.4 |         425.8 |            83.2 |                   80.5 |                5.12 |               15.62 |                 15.3  |           37.7 |             48   |                7631 |                           nan |
| position   | snapkv1024 |          150 |           61.1 |             66.7 |                   109.1 |         425.8 |            64.4 |                   84.9 |                6.61 |               15.62 |                 15.27 |           37.7 |             35.4 |                7631 |                           100 |
| position   | snapkv2048 |          150 |           61.1 |             61.1 |                   100   |         425.8 |           120.3 |                   71.7 |                3.54 |               15.62 |                 15.32 |           37.7 |             35.6 |                7631 |                           100 |

## Position test: pass@1 by depth of the relevant code

| method     |    0 |   25 |   50 |   75 |   100 |
|:-----------|-----:|-----:|-----:|-----:|------:|
| full       | 61.1 | 44.4 | 66.7 | 61.1 |  72.2 |
| kivi2      | 61.1 | 50   | 50   | 61.1 |  72.2 |
| kivi4      | 55.6 | 55.6 | 61.1 | 55.6 |  77.8 |
| snapkv1024 | 66.7 | 55.6 | 61.1 | 72.2 |  77.8 |
| snapkv2048 | 61.1 | 44.4 | 61.1 | 61.1 |  77.8 |

## Charts

![](quality_vs_memory.png)
![](position_by_depth.png)

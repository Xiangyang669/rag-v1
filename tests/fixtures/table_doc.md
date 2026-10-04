---
title: 检索参数速查
---

# 检索参数速查

本文汇总三层检索系统里可以调的参数。表格是结构化的检索单元——
它独立成块，不会被定长切片从中间截断。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| k | 10 | 每路召回条数 |
| rrf_k | 60 | RRF 的平滑常数 |

上表是短表，整张能放进一个块。下面是长表，会被按「表头 + 若干行」切开，
但**每一块都会重复表头**，否则后面那些行就只是无意义的裸值。

## 全部参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| k | int | 10 | 每一路召回的条数上限，融合前生效 |
| rrf_k | int | 60 | RRF 的平滑常数，越大越削弱头部排名差异 |
| max_chars | int | 1200 | 单块字符上限，超过的节会走 OVER_CAP 再切 |
| vector_enabled | bool | true | 是否启用向量路，关掉后图里不会有该节点 |
| fts_enabled | bool | true | 是否启用全文路，其路径名是 fulltext |
| graph_enabled | bool | true | 是否启用图谱路，只做一跳遍历 |
| hnsw_space | str | cosine | 向量库的距离空间，改它要重建索引 |
| bm25_weight | float | 1.0 | 全文路打分权重，仅影响单路排序不影响融合 |
| entity_alias | bool | true | 是否启用别名表做实体消解 |
| chunk_overlap | int | 0 | 保留参数，当前版本不做重叠切分 |
| tokenize_mode | str | unicode61 | 全文索引的分词器，中文需配合 jieba 预处理 |
| rerank_enabled | bool | false | 重排开关，本版本没有重排模块因此无效 |
| graph_hops | int | 1 | 图谱遍历跳数，大于一是后续版本的规划 |
| embedding_model | str | bge-m3 | 向量模型名，换模型必须全量重建索引 |
| embedding_dim | int | 1024 | 向量维度，与模型绑定，写错会在入库时报错 |
| batch_size | int | 64 | 入库时的批量大小，只影响速度不影响结果 |
| timeout_seconds | int | 30 | 单次外部调用的超时，包含 embedding 与图谱抽取 |
| retry_times | int | 2 | 外部调用失败后的重试次数，指数退避 |
| cache_ttl | int | 3600 | 查询结果缓存的有效期，零表示不缓存 |
| log_level | str | info | 日志级别，debug 会打印每路的耗时明细 |
| vector_top_k | int | 10 | 向量路召回条数，与全局 k 分开设置时以本参数为准 |
| fts_top_k | int | 10 | 全文路召回条数，单独调它只影响该路的候选集大小 |
| graph_top_k | int | 10 | 图谱路召回条数，命中数通常远小于另外两路 |
| min_score | float | 0.0 | 单路最低分数阈值，零表示不过滤，融合前生效 |
| dedup_enabled | bool | true | 是否按 chunk_id 去重，三路同时命中时只保留一份 |
| trace_enabled | bool | false | 是否打 Langfuse 埋点，没有 key 时自动退化为空操作 |

## 注意事项

改上面任何一个参数之后，如果它影响块边界或向量内容，都必须**全量重建索引**。
部分参数的改动只影响查询期，可以热改；哪些参数属于哪一类，见参数说明。

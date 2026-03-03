# Milvus Hybrid Collection 设计

目标：支持聊天驱动选品（BM25 sparse + HNSW dense）和可溯源的事实对比。

## 1) `policy_chunks_hybrid`

用途：
- 主检索库。用户提问“推荐哪个计划/找更匹配计划”时，先在 chunks 里做 hybrid 检索召回候选计划。

主键与元数据字段：
- `chunk_id` (VARCHAR, PK)
- `plan_id` (VARCHAR)
- `plan_name` (VARCHAR)
- `section_path` (VARCHAR)
- `page_start` (INT64)
- `page_end` (INT64)
- `source_ref` (VARCHAR)
- `language` (VARCHAR)
- `text` (VARCHAR)
- `created_at` (INT64)

向量字段：
- `dense_embedding` (FLOAT_VECTOR, dim=embedding_dim)
- `sparse_embedding` (SPARSE_FLOAT_VECTOR)

索引：
- `dense_embedding`: `HNSW`, metric `COSINE`, params `{M:16, efConstruction:200}`
- `sparse_embedding`: `SPARSE_INVERTED_INDEX`, metric `IP`, params `{drop_ratio_build:0.1}`

## 2) `policy_facts_hybrid`

用途：
- 结构化事实层。用于对比解释、引用来源、后续事实级检索与审计。

主键与元数据字段：
- `fact_id` (VARCHAR, PK)
- `plan_id` (VARCHAR)
- `plan_name` (VARCHAR)
- `dimension_key` (VARCHAR)
- `dimension_label` (VARCHAR)
- `value_text` (VARCHAR)
- `normalized_value` (VARCHAR)
- `unit` (VARCHAR)
- `condition_text` (VARCHAR)
- `applicability` (VARCHAR)
- `source_chunk_id` (VARCHAR)
- `source_page` (INT64)
- `source_section` (VARCHAR)
- `confidence` (FLOAT)
- `created_at` (INT64)

向量字段：
- `dense_embedding` (FLOAT_VECTOR, dim=embedding_dim)
- `sparse_embedding` (SPARSE_FLOAT_VECTOR)

索引：
- `dense_embedding`: `HNSW`, metric `COSINE`, params `{M:16, efConstruction:200}`
- `sparse_embedding`: `SPARSE_INVERTED_INDEX`, metric `IP`, params `{drop_ratio_build:0.1}`

## 为什么要两张表

- `chunks` 解决“召回”问题：文档大、语义分布广，适合做产品发现。
- `facts` 解决“解释”问题：维度化结构清晰，便于对比表与来源溯源。
- 聊天链路：先用 `chunks` 找计划，再用 `facts` 和 SQLite 会话状态刷新对比。

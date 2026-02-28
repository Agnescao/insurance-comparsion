# Milvus Collection 设计

本文档对应原型中的两类核心 collection：`policy_chunks` 与 `policy_facts`，用于支持“高召回检索 + 精准并列比较 + 溯源”。

## 1) `policy_chunks`

用途:
- 存储可检索文本块（chunk）
- 支持聊天问题的语义召回
- 保留页码、章节、来源引用做证据回溯

建议字段:
- `chunk_id` (VARCHAR, PK)
- `plan_id` (VARCHAR, scalar index)
- `plan_name` (VARCHAR)
- `product_version` (VARCHAR)
- `section_path` (VARCHAR)
- `page_start` (INT64)
- `page_end` (INT64)
- `source_ref` (VARCHAR)
- `language` (VARCHAR)
- `text` (VARCHAR)
- `created_at` (INT64)
- `embedding` (FLOAT_VECTOR, dim=256 或在线模型维度)

索引:
- 向量索引: `embedding` 使用 `AUTOINDEX` + `COSINE`
- 标量过滤: `plan_id`、`language`

检索模式:
- 先按 `plan_id in (...)`、`language='zh'` 过滤
- 再做 top-k 向量检索
- 返回 chunk + `source_ref/page/section` 用于回答与界面溯源

## 2) `policy_facts`

用途:
- 存储结构化事实，直接驱动对比表格
- 支持维度级查询、差异高亮、规则计算

建议字段:
- `fact_id` (VARCHAR, PK)
- `plan_id` (VARCHAR, scalar index)
- `dimension_key` (VARCHAR, scalar index)
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
- `embedding` (FLOAT_VECTOR)

索引:
- 向量索引: `embedding` + `COSINE`
- 标量过滤: `plan_id`, `dimension_key`, `confidence`

查询模式:
- 对比表: `WHERE plan_id IN (...) AND dimension_key IN (...)`，按 `confidence desc` 取每计划每维度最佳值
- 聊天补充检索: 先按 `dimension_key` 和 `condition_text` 过滤，再做向量重排

## 3) 写入策略

- `policy_chunks`: 每个 chunk 写一条，保留完整来源定位
- `policy_facts`: 每个事实写一条，保留 `source_chunk_id` 与 `source_page`
- 保证幂等: 以 `plan_id` 为粒度重建（先删旧再写新）

## 4) 与 SQLite 的关系

原型中 SQLite 是在线服务的主读库（低运维成本），Milvus 为可选增强层:
- 表格查询优先走 SQLite `policy_facts`
- 聊天语义检索可同时利用 Milvus `policy_chunks/policy_facts`
- 两者通过 `plan_id/chunk_id/fact_id` 关联

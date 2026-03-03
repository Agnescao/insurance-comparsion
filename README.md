## 项目demo
[点击查看 Demo 视频](leicao-demo-insurancecomparsion.mp4)


# 保险产品比较工具原型

本项目提供一个可本地运行的工作原型，覆盖：
- 多计划并列比较表（差异高亮）
- 每个字段来源定位（页码/section）
- 聊天面板连续对话（保存已选计划与维度）
- 聊天可触发选品检索（Hybrid: Sparse + Dense）
- PDF 解析 -> chunks/facts -> 入库流程

## 技术栈

- 前端: React + Tailwind + Vite
- 后端: FastAPI + SQLAlchemy
- 数据库: SQLite（业务态）+ Milvus（检索态 HNSW+BM25）
- PDF 解析: PyMuPDF（可扩展接入OCR）
- 向量embedding: text-embedding-v3
- 结构化提取: qwen-turbo +基于规则的后处理
- LLM: DashScope OpenAI 兼容接口（DeepSeek）聊天

## 项目结构

```
insurance-comparison/
├── backend/                      # 后端服务
│   ├── app/                      # FastAPI 应用核心
│   │   ├── services/             # 业务服务层
│   │   │   ├── chat.py           # 聊天会话管理
│   │   │   ├── chunking.py       # 文档分块策略
│   │   │   ├── compare.py        # 比较引擎
│   │   │   ├── dimensions.py     # 维度定义
│   │   │   ├── embeddings.py     # 向量生成
│   │   │   ├── fact_extractor.py # 事实抽取器
│   │   │   ├── hybrid_retriever.py # 混合检索器
│   │   │   ├── ingestion.py      # 数据入库管线
│   │   │   ├── llm_fact_extractor.py # LLM 事实抽取
│   │   │   ├── llm_planner.py    # LLM 意图规划
│   │   │   ├── milvus_hybrid_store.py # Milvus 混合存储
│   │   │   ├── parser.py         # PDF 解析器
│   │   │   └── sparse_bm25.py    # BM25 稀疏向量
│   │   ├── config.py             # 配置管理
│   │   ├── database.py           # 数据库连接
│   │   ├── main.py               # FastAPI 入口
│   │   ├── models.py             # SQLAlchemy 模型
│   │   └── schemas.py            # Pydantic 模式
│   ├── data/                     # 后端数据目录
│   │   └── retrieval_eval.sample.jsonl # 检索评测集
│   ├── scripts/                  # 运维脚本
│   │   ├── eval_retrieval_hitk.py # Hit@K 评测
│   │   └── ingest.py             # 手动入库脚本
│   ├── .env.example              # 环境变量示例
│   └── requirements.txt          # Python 依赖
├── data/                         # 产品数据源
│   ├── output/                   # 解析输出产物
│   │   └── <plan_name>/          # 每个计划的解析结果
│   │       ├── chunks.json       # 分块数据
│   │       ├── document.md       # Markdown 文档
│   │       └── facts.json        # 结构化事实
│   └── *.pdf                     # 原始保险产品 PDF
├── frontend/                     # 前端应用
│   ├── src/                      # React 源代码
│   │   ├── components/           # UI 组件
│   │   │   ├── ChatPanel.jsx     # 聊天面板
│   │   │   ├── CompareTable.jsx  # 比较表格
│   │   │   └── PlanDimensionPanel.jsx # 计划/维度选择器
│   │   ├── App.jsx               # 主应用组件
│   │   ├── api.js                # API 客户端
│   │   └── main.jsx              # 入口文件
│   ├── index.html                # HTML 模板
│   ├── package.json              # npm 配置
│   └── vite.config.js            # Vite 配置
├── docs/                         # 设计文档
│   └── architecture.md           # 系统架构
├── README.md                     # 项目说明
└── 需求文档.md                    # 详细需求
```

## 快速启动

### 1) 启动后端

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="."
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

说明：
- 默认不在 startup 自动 ingest
- 手动触发: `POST /api/ingest/run`

### 2) 启动前端

```powershell
cd frontend
npm install
npm run dev
```



默认访问：`http://localhost:5173`（若占用会自动切到 5174/5175）

### milvus 云服务启动

# 1、启动Milvus数据库的docker容器
sudo docker compose up -d
docker restart milvus-standalone

# 2 验证Milvus数据库是否成功启动
docker ps

## 环境变量

参考 `backend/.env.example`：
- `EMBEDDING_PROVIDER=hash|qwen`
- `QWEN_API_KEY=...`（当 provider=qwen 时使用）
- `MILVUS_ENABLED=true|false`
- `MILVUS_URI=tcp://<host>:19530`
- `LLM_ENABLED=true|false`
- `LLM_API_KEY=...`（DashScope key）
- `LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`
- `LLM_PLANNER_MODEL=deepseek-v3`
- `LLM_ANSWER_MODEL=deepseek-v3`


## API 概览

- `GET /api/plans`: 获取计划列表
- `GET /api/dimensions`: 获取标准比较维度
- `POST /api/compare`: 按计划/维度生成比较表
- `POST /api/chat/session`: 创建会话与初始状态
- `POST /api/chat/message`: 聊天提问并更新上下文状态
- `POST /api/chat/message/stream`: 聊天流式输出（SSE）
- `POST /api/ingest/run`: 重跑入库管线

## 检索评测

你可以用离线标注集评测命中率（Hit@K）：

```powershell
$env:PYTHONPATH="backend"
python backend/scripts/eval_retrieval_hitk.py --dataset backend/data/retrieval_eval.jsonl --k 3
```

`JSONL` 每行格式：
```json
{"query":"卵巢癌保障更好的计划","relevant_plan_ids":["<plan_id_1>","<plan_id_2>"]}
```

## 与需求对应

- 并列比较表: 已实现，支持至少 2 个计划
- 关键维度: 覆盖住院/门诊/手术/保费/限额/自付/附加险/除外责任
- 来源引用: 每个单元格显示 `section + 页码`
- 差异高亮: 行级和单元格级高亮
- 聊天连续上下文: `chat_sessions/chat_turns/session_state` 持久化
- 聊天动态加维度: 例如“卵巢癌场景比较”
- 聊天触发选品检索: 支持 hybrid 检索后更新 `selected_plans`

## 当前边界

- pbf本地解析为主，复杂扫描件建议接入 dots-ocr/layout 模型
- 10 秒 SLA 在当前样本规模可达，规模扩大需缓存与并发优化

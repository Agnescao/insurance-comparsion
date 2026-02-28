# 保险产品比较工具原型

本项目提供一个可本地运行的工作原型，覆盖:
- 多计划并列比较表（差异高亮）
- 每个字段来源定位（页码/section）
- 聊天面板连续对话（保存已选计划与维度）
- PDF 解析 -> chunks/facts -> 入库流程

## 技术栈

- 前端: React + Tailwind + Vite
- 后端: FastAPI + SQLAlchemy
- 数据库: SQLite（必选）+ Milvus（可选增强）
- PDF 解析: PyMuPDF
- 向量: 默认 Hash Embedding（离线可跑），可切换 Qwen Embedding

## 项目结构

- `backend/app/main.py`: FastAPI 入口
- `backend/app/models.py`: SQLite 表模型（plans/chunks/facts/chat）
- `backend/app/services/ingestion.py`: 解析与入库管线
- `backend/app/services/milvus_store.py`: Milvus collection 建模与写入
- `frontend/src/App.jsx`: 三栏仪表板
- `docs/milvus_collection_design.md`: collection 结构设计说明

## 快速启动

### 1) 启动后端

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="."
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

说明:
- 首次启动会自动扫描 `../data/*.pdf` 并入库
- 也可手动触发: `POST /api/ingest/run`

http://localhost:8000/api/plans
### 2) 启动前端

```powershell
cd frontend
npm install
npm run dev
```


默认访问: `http://localhost:5173`

## 环境变量

参考 `backend/.env.example`:
- `EMBEDDING_PROVIDER=hash|qwen`
- `QWEN_API_KEY=...` (当 provider=qwen 时使用)
- `MILVUS_ENABLED=true|false`
- `MILVUS_URI=tcp://121.41.85.215:19530`

## API 概览

- `GET /api/plans`: 获取计划列表
- `GET /api/dimensions`: 获取标准比较维度
- `POST /api/compare`: 按计划/维度生成比较表
- `POST /api/chat/session`: 创建会话与初始状态
- `POST /api/chat/message`: 聊天提问并更新上下文状态
- `POST /api/ingest/run`: 重跑入库管线

## 与需求对应

- 并列比较表: 已实现，至少支持2个计划
- 关键维度: 已覆盖住院/门诊/手术/保费/限额/自付/附加险/除外责任
- 来源引用: 每个单元格显示 `section + 页码`
- 差异高亮: 行级和单元格级高亮
- 聊天连续上下文: `chat_sessions/chat_turns/session_state` 持久化
- 聊天动态加维度: 例如 “加入卵巢癌场景比较”

## 当前原型边界

- OCR 在线模型与 Qwen 向量已预留接口，默认使用本地抽取与本地向量降级
- 事实抽取为规则版（regex/关键词），生产环境建议替换为 LLM 抽取器
- 10 秒 SLA 在本地样本规模下可达，超大规模时需要缓存和并发优化

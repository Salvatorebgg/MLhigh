# 二次集成说明

## 架构概述

本平台采用 FastAPI + 原生前端 SPA 的轻量架构：

- **后端**：Python FastAPI，负责上传、读取、变量识别、高级统计、机器学习、三线表和导出。
- **前端**：HTML / CSS / JavaScript，无前端框架依赖。
- **图表**：本地 Plotly.js，图表配置集中在 `app/services/chart_service.py`，渲染后统一经过 CNS 出版级后处理。
- **数据链路**：先选方法，再加载对应示例或上传数据；分析通过 `/api/analyze` 进行全流程计算。

## 推荐集成方式

### 方式一：独立部署

```bash
python run.py
```

默认访问：

```text
http://127.0.0.1:8868
```

主站可以用 iframe 或普通链接集成：

```html
<iframe src="http://127.0.0.1:8868" width="100%" height="860"></iframe>
<a href="http://127.0.0.1:8868" target="_blank" rel="noreferrer">打开临床高级统计平台</a>
```

如果默认端口被占用，可指定新端口：

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8872
```

### 方式二：挂载到 FastAPI 主应用

```python
import sys
from fastapi import FastAPI

sys.path.insert(0, "/path/to/MLhigh")
from app.main import app as mlhigh_app

main_app = FastAPI()
main_app.mount("/mlhigh", mlhigh_app)
```

访问：

```text
http://your-site.com/mlhigh
```

### 方式三：只调用 API

上传文件：

```javascript
const formData = new FormData();
formData.append("file", file);

const uploadResp = await fetch("http://127.0.0.1:8868/api/upload", {
  method: "POST",
  body: formData,
});
const upload = await uploadResp.json();
```

运行分析：

```javascript
const analyzeResp = await fetch("http://127.0.0.1:8868/api/analyze", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    method_id: "gee",
    use_demo: true,
    dataset_name: "gee_example",
    params: { group_var: "arm", time_var: "time", outcome_var: "sbp" },
  }),
});
const result = await analyzeResp.json();
// result = { tables: [...], charts: [...], diagnostics: [...], discussion: "..." }
```

生成基线资料表：

```javascript
const tableResp = await fetch("http://127.0.0.1:8868/api/table/baseline", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    upload_id: upload.upload_id,
    group_var: "arm",
    variables: ["age", "sex", "bmi"],
    decimal_places: 2,
    p_digits: 3,
  }),
});
const table = await tableResp.json();
```

### 方式四：复用前端方法配置

如果主项目已有自己的前端，只想复用分析方法：

1. 引入 Plotly.js。
2. 复制或打包 `app/static/js/chartThemes.js`。
3. 使用 `/api/dataset/data` 获取完整列式数据。
4. 使用 `/api/analyze` 运行分析。
5. 使用返回的 charts 数据通过 Plotly 渲染。

## 关键接口

| 端点 | 用途 |
|---|---|
| `/api/upload` | 上传文件并返回预览、变量类型和摘要 |
| `/api/read-sheet` | 切换 Excel 工作表 |
| `/api/examples` | 获取示例数据列表 |
| `/api/examples/{name}` | 获取示例预览和变量类型 |
| `/api/dataset/data` | 获取完整列式数据 |
| `/api/methods` | 获取方法目录 |
| `/api/analyze` | 一键运行分析方法 |
| `/api/table/baseline` | 生成基线三线表 |
| `/api/table/descriptive` | 生成描述统计表 |
| `/api/table/missing` | 生成缺失统计表 |

## CORS 配置

默认允许任意来源调用。生产环境建议限制来源：

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 生产部署

### Uvicorn

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8868
```

### Gunicorn + Uvicorn Worker

```bash
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8868
```

### Nginx 反向代理

```nginx
server {
    listen 80;
    server_name mlhigh.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8868;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 200M;
    }
}
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8868
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8868"]
```

## 数据和会话隔离

上传文件保存在 `data/uploads/`，文件名前缀使用 UUID。生产环境建议：

- 定时清理 `data/uploads/` 和 `outputs/`。
- 在外层应用中绑定用户身份和 `upload_id`。
- 对上传大小、并发请求和导出频率做限制。
- 如需长期保存结果，使用数据库记录上传、分析参数和导出文件路径。

## 注意事项

- 前端默认加载本地 `app/static/vendor/plotly.min.js`，避免 CDN 失败导致图表渲染或 PNG/SVG 导出不可用。
- 图表区 PNG/SVG 使用 `Plotly.toImage` 导出当前画布；CSV 优先导出当前分析所选变量对应的完整源数据。
- 出版级导出调用 `/api/export/chart/publication` 生成 matplotlib/seaborn PNG/SVG/PDF。
- 分析结果渲染推荐使用 `renderAllCharts` 和 `renderResultTables`，不要直接操作 DOM。
- 新增方法时必须同时维护示例数据、`exampleDataset` 和默认参数映射，确保用户点击方法即可得到可运行示例。

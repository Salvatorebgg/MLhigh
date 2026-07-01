# MLhigh 临床统计分析平台

面向临床科研数据的统计分析与机器学习工具。当前版本采用“方法优先”的工作流：用户先选择统计方法，再上传自己的数据或加载该方法对应的示例数据，随后选择变量、调节参数并运行分析。

## 快速启动

```bash
python run.py
```

默认访问地址：

```text
http://127.0.0.1:28872
```

也可以直接指定端口：

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 28872
```

## 当前工作流

1. 方法选择：先选择本次要做的统计分析方法。
2. 数据上传：上传 CSV/Excel，或加载该方法绑定的独立示例数据。
3. 变量选择：选择研究变量、协变量和结局变量，并调节该方法的参数。
4. 结果生成：输出图表、结果表格和自动解释。

## 方法与示例

- 高级统计：13 个方法。
- 机器学习：13 个方法，包含原来的特征工程、多模型比较、降维分析和聚类分析。
- 示例数据：26 个方法各自绑定 1 个独立示例数据，不再提供综合示例数据。
- 可调参数：每个方法都有独立参数配置，包括阈值、训练集比例、随机种子、模型结构、迭代次数、Bootstrap 次数等。

## 前端结构

当前前端已简化为核心文件：

```text
app/static/
├── index.html          # 四步式主界面
├── styles.css          # 蓝白浅灰工具台风格
├── vendor/
│   └── plotly.min.js
└── js/
    ├── app.js          # 方法选择、数据加载、变量/参数、运行分析
    ├── utils.js        # 全局状态和通用工具
    ├── chartThemes.js  # 图表主题
    └── charts.js       # Plotly 图表渲染
```

已删除旧的前端拆分脚本：`methodConfigs.js`、`upload.js`、`variableSelect.js`、`dataPreview.js`、`tableGenerator.js`、`download.js`。

## 后端结构

```text
app/
├── main.py                  # FastAPI 路由、方法目录、参数配置
├── schemas.py               # 请求模型
├── config.py                # 路径配置
└── services/
    ├── io_service.py        # 上传和示例数据读取
    ├── sample_service.py    # 26 个方法示例数据生成器
    ├── stats_service.py     # 高级统计方法
    ├── ml_service.py        # 机器学习方法
    ├── chart_service.py     # 图表生成
    ├── table_service.py     # 表格生成
    ├── report_service.py    # 结果解释/报告
    └── export_service.py    # 导出服务
```

## 常用 API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/api/health` | GET | 健康检查 |
| `/api/methods` | GET | 获取方法目录与参数配置 |
| `/api/examples` | GET | 获取示例数据列表 |
| `/api/examples/{name}` | GET | 获取示例数据预览 |
| `/api/examples/{name}/download` | GET | 下载示例 CSV |
| `/api/upload` | POST | 上传数据文件 |
| `/api/analyze` | POST | 运行指定分析方法 |
| `/api/analyze/report` | POST | 生成分析报告 |

## 验证

```bash
python tests\smoke.py
python -m py_compile app\main.py app\services\sample_service.py run.py
node --check app\static\js\app.js
node --check app\static\js\charts.js
```

## 文档

- [二次集成说明](docs/INTEGRATION.md)
- [扩展说明](docs/EXTENSION.md)

## 许可证

MIT License

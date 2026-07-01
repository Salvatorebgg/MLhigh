# 后续扩展说明

本文档说明如何添加新分析方法、新示例数据、新主题和新表格类型。当前项目的关键约定是：**每个分析方法必须有对应示例数据，并且前端参数配置要能自动填入默认值**。

## 添加新分析方法

### 1. 准备示例数据

**方式 A（推荐）：自动生成**

在 `app/services/sample_service.py` 中添加生成器函数：

```python
def make_new_method_example() -> pd.DataFrame:
    n = 200
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.normal(55, 12, n).astype(int),
        "bmi": np.random.normal(25, 4, n).round(1),
        "treatment": np.random.choice(["Drug A", "Drug B", "Placebo"], n),
        "outcome": np.random.binomial(1, 0.4, n),
        "biomarker": np.random.normal(100, 20, n).round(1),
    })
```

注册到 `EXAMPLE_MAKERS`：

```python
EXAMPLE_MAKERS = {
    # existing examples
    "new_method_example": make_new_method_example,
}
```

应用启动时会在缺失时自动生成 `data/examples/new_method_example.csv`。

**方式 B：手工构造数据文件**

如果方法需要真实格式数据（如 LDSC 遗传共病分析），直接将 CSV 文件放入 `data/examples/` 目录，无需注册 `EXAMPLE_MAKERS`。需确保文件编码为 UTF-8 BOM。

### 2. 添加后端分析逻辑

在 `app/services/stats_service.py`（统计方法）或 `app/services/ml_service.py`（ML/工具方法）中添加分析函数：

```python
def run_new_method(df: pd.DataFrame, **params) -> dict:
    """New analysis method implementation."""
    # 计算逻辑
    tables = []
    charts = []
    diagnostics = []

    # 返回统一结构
    return {
        "tables": tables,
        "charts": charts,
        "diagnostics": diagnostics,
        "discussion": "分析讨论文本...",
    }
```

然后将函数注册到对应路由字典：
- 统计方法 → `STATS_ROUTER`（`stats_service.py` 末尾）
- ML/工具方法 → `ML_ROUTER`（`ml_service.py` 末尾）

```python
STATS_ROUTER["new_method"] = run_new_method   # 或 ML_ROUTER
```

### 3. 注册方法到 METHOD_CATALOG

在 `app/main.py` 的 `METHOD_CATALOG` 中添加：

```python
"new_method": {
    "id": "new_method",
    "name": "新分析方法",
    "category": "advanced_stats",  # advanced_stats | ml_models | integrated_tools
    "description": "方法说明文字",
    "example_dataset": "new_method_example",
    "default_params": {
        "group_var": "treatment",
        "outcome_var": "outcome",
    },
},
```

### 4. 前端展示与参数配置

前端方法列表直接来自 `/api/methods`，不再维护独立的 `methodConfigs.js`。新增方法时，优先在 `app/main.py` 的 `METHOD_CATALOG` 中写清楚：

- `name`、`icon`、`category`、`description`
- `example_dataset`
- `params` 中每个可调参数的 `key`、`label`、`type`、`default`、`options`

如果新增了特殊控件类型，再到 `app/static/js/app.js` 的 `renderParamField()` 中补充渲染逻辑。普通的 `select`、`multi_select`、`number`、`checkbox`、`text` 会自动渲染。

## 添加新图表主题

在 `app/static/js/chartThemes.js` 的 `CHART_THEMES` 中添加新主题对象：

```javascript
myNewTheme: {
  name: "我的主题",
  fontFamily: "'Noto Sans SC', 'Microsoft YaHei', sans-serif",
  bgColor: "#ffffff",
  plotBgColor: "#ffffff",
  gridColor: "rgba(0,0,0,0.06)",
  zeroLineColor: "rgba(0,0,0,0.15)",
  titleColor: "#1a1a1a",
  axisColor: "#1a1a1a",
  axisLineColor: "#1a1a1a",
  ink: "#1a1a1a",
  colorway: ["#2E6F9E", "#D95F59", "#2A9D8F", "#E9A93A", "#6F5AA7"],
  markerLine: "#ffffff",
  opacity: 0.90,
  titleFontSize: 17,
  axisFontSize: 12,
  tickFontSize: 11,
  legendFontSize: 11,
},
```

在 `app/static/index.html` 的主题选择器中添加对应选项。

## 添加新的三线表类型

### 1. 后端表格函数

在 `app/services/table_service.py` 中添加：

```python
def build_my_table(df: pd.DataFrame, variables: list[str] | None = None) -> dict:
    rows = []
    # 计算逻辑
    return {
        "columns": ["Variable", "Value"],
        "rows": rows,
        "n_total": len(df),
    }
```

### 2. FastAPI 路由

在 `app/main.py` 中添加：

```python
@app.post("/api/table/my-table")
def my_table(req: TableRequest) -> dict:
    df = _get_df(req)
    return build_my_table(df, req.variables)
```

### 3. 前端入口

如需在当前简化界面中增加入口，在 `app/static/index.html` 中添加按钮，并在 `app/static/js/app.js` 中添加对应请求逻辑。

## 添加新数据格式

编辑 `app/services/io_service.py`：

1. 在 `SUPPORTED_EXTENSIONS` 中添加后缀。
2. 在 `read_file()` 中添加读取逻辑。
3. 返回前调用 `normalize_dataframe(df)`，保持缺失值、空行和列名处理一致。

## UI 扩展规则

- 方法选择列表默认单列纵向排列，新增方法无需改 CSS。
- 每个方法卡片应提供 `name`、`description`、`icon`、`category` 和 `exampleDataset`。
- 不要让分析依赖 `STATE.previewRows`；预览只用于展示，分析应使用 `/api/dataset/data` 接口获取完整数据。
- 新增方法后，`chart_service.py` 可能需要添加对应的图表生成函数。
- 出版级导出由 `chart_service.py` 中的 matplotlib 函数处理；如需高质量 PDF/TIFF 导出，确保对应方法有 matplotlib 图表生成器。
- 需要新图表类型时，在 `chartThemes.js` 中可选添加主题，所有图表渲染自动继承当前选定主题。

## 验证命令

每次新增方法或接口后建议运行：

```bash
# Python 端验证
python tests\smoke.py                      # 示例数据生成、变量识别、表格服务
python -m compileall app                   # 语法检查

# 前端 JS 语法检查
node --check app\static\js\charts.js
node --check app\static\js\app.js
```

也可以启动临时端口验证新接口：

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 28872
```

```bash
# 验证分析方法
curl -X POST http://127.0.0.1:28872/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"method_id": "new_method", "use_demo": true, "dataset_name": "new_method_example", "params": {"group_var": "treatment"}}'

# 验证健康检查
curl http://127.0.0.1:28872/api/health

# 验证方法目录
curl http://127.0.0.1:28872/api/methods
```

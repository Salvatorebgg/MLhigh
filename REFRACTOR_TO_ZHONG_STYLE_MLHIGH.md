# MLhigh → jichutongji_zhong 风格重构说明

## 重构目标

以 `jichutongji_zhong` 项目的两栏式工作站界面为基准，在不改动 MLhigh 统计分析、机器学习建模、绘图生成、下载导出、结果表格、诊断图和结果讨论等核心功能逻辑的前提下，统一 MLhigh 的界面观感。

## 已完成内容

### 1. 三栏改为两栏

原 MLhigh：
- 左侧：方法和数据
- 中间：结果工作区
- 右侧：参数和外观

现已改为：
- 左侧：仅保留步骤导航
- 右侧：全部功能工作区

左侧步骤包括：
1. 分析方法
2. 上传数据
3. 参数配置
4. 执行分析
5. 数据概览
6. 结果可视化
7. 结果表格
8. 诊断评估
9. 结果讨论

### 2. 功能全部移动到右侧

右侧工作区按步骤展示：
- 分析方法：方法分类 + 方法卡片
- 上传数据：加载示例、上传文件、示例下载
- 参数配置：变量映射、模型参数、图形外观
- 执行分析：运行分析按钮
- 数据概览：样本量、变量数、缺失率、数据预览
- 结果可视化：图形预览与下载
- 结果表格、诊断评估、结果讨论：保持原 MLhigh 结果逻辑

### 3. 视觉风格靠拢 jichutongji_zhong

统一了：
- 蓝白主色调
- 左侧步骤式导航
- 白底卡片
- 浅灰边框
- 蓝色激活态
- 按钮圆角、阴影、字体大小和字重
- 方法分类按钮
- 方法卡片
- 下载菜单样式
- 表格、图形、结果区卡片风格

### 4. 核心功能逻辑保持不变

未改动：
- `ml_service.py`
- `stats_service.py`
- `chart_service.py`
- `sample_service.py`
- `table_service.py`
- `/api/analyze`
- `/api/export/*`
- 方法配置和绘图数据结构

前端只做了：
- `index.html` 布局重构
- `styles.css` 追加 zhong 风格样式
- `app.js` 追加两栏工作流导航适配

## 已检查

- 前端主要 JS 语法检查通过：
  - app.js
  - charts.js
  - download.js
  - upload.js
  - variableSelect.js
  - tableGenerator.js
  - methodConfigs.js
- 后端 `app/main.py` 编译通过
- `tests/smoke.py` 通过

## 注意

本版保留 MLhigh 的分析、绘图和报告生成逻辑，只改变界面结构和视觉风格。

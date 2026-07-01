# MLhigh v12：修复岭回归图表渲染失败

## 本轮修复

1. 修复“图表渲染失败”
   - v11 中图形布局函数存在覆盖/递归风险，导致前端 Plotly 渲染失败；
   - v12 改为独立的最终布局函数，不再递归调用旧函数；
   - 保留 jichutongji_zhong 风格的右侧参数调节区。

2. 修复 Plotly JSON 数据格式
   - 后端原先使用 `fig.to_dict() + json.dumps(default=str)`，会把 numpy 数组转成字符串，或输出 Plotly 6 的 `{dtype,bdata}` 二进制数组；
   - 部分浏览器端 Plotly 无法稳定渲染这种格式；
   - v12 新增 `_plotly_json_safe()`，把 numpy / pandas / typed-array 全部转换成普通 JSON list；
   - 解决岭回归 Lasso 图中 x/y 数据不合法导致渲染失败的问题。

3. 验证
   - 使用你截图中的变量组合：
     - 研究变量：baseline_bmi、cholesterol
     - 协变量：crp、dbp、sbp
     - 结局变量：outcome_continuous
     - 方法：Lasso/岭回归
   - 后端返回 200；
   - 图表 JSON 中 x/y 均已转换为普通数组；
   - 前端 JS 语法检查通过；
   - 后端 Python 编译通过。

## 说明

本轮重点不是继续调整界面，而是先修复你截图中“能运行但图表渲染失败”的核心问题，保证可运行方法的图形可以正常显示。

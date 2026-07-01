# MLhigh v13：Lasso 回归与岭回归分离

## 本轮修改

1. 方法层面拆分
   - 原来的“Lasso/岭回归”拆分为两个独立方法：
     - Lasso 回归
     - 岭回归

2. 统计逻辑区别
   - Lasso 回归：L1 正则化，可以把部分系数压缩为 0，适合变量筛选和稀疏建模。
   - 岭回归：L2 正则化，主要缩小系数、缓解多重共线性，通常不会把变量系数压缩为 0。

3. 后端路由
   - 新增 `ml_ridge`；
   - `ml_lasso` 固定使用 `regularization = lasso`；
   - `ml_ridge` 固定使用 `regularization = ridge`；
   - 两者在方法列表、后端校验、自动参数推导、运行分析中均为独立方法。

4. 前端校验
   - `/api/methods` 返回两个独立方法；
   - `/api/validate-methods` 会分别返回 `ml_lasso` 和 `ml_ridge` 的可用性；
   - 用户不再需要在参数中切换 lasso/ridge。

## 已验证

- `/api/methods` 中同时存在 `ml_lasso` 与 `ml_ridge`；
- 同一组连续结局变量下，Lasso 回归可运行；
- 同一组连续结局变量下，岭回归可运行；
- Lasso 回归结果标题为“Lasso 回归结果”；
- 岭回归结果标题为“岭回归结果”；
- 前端 JS 语法通过；
- 后端 Python 编译通过。

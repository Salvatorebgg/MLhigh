const APP = {
  methods: [],
  methodMap: new Map(),
  activeResultTab: 'chart',
  sizeLinked: true,
  chartAspect: 760 / 600,
  lastResult: null,
  lastResultTables: [],
  validationToken: 0,
};

function dom(id) {
  return document.getElementById(id);
}

function domAll(selector, parent = document) {
  return Array.from(parent.querySelectorAll(selector));
}

function html(value) {
  return escapeHtml(value ?? '');
}

const PARAM_CN_LABELS = {
  split_ratio: '训练集比例',
  random_state: '随机种子',
  cov_struct: '相关结构',
  max_iter: '最大迭代次数',
  scale: '尺度参数',
  caliper: '匹配卡钳',
  matching_ratio: '匹配比例',
  ps_model: '倾向评分模型',
  standardize: '协变量标准化',
  missing_strategy: '缺失值处理',
  alpha: 'Alpha 正则化强度',
  alpha_mode: 'Alpha 选择方式',
  bootstrap: 'Bootstrap 次数',
  delta_range: '扰动范围',
  effect_model: '效应模型',
  n_estimators: '树或基学习器数量',
  min_samples_leaf: '叶节点最小样本数',
  trim_quantile: '极端值截尾比例',
  time_horizon: '预测时间点',
  ties_method: '并列时间处理',
  reference_group: '参考组',
  n_cycles: '模拟周期数',
  cycle_length: '周期长度',
  discount_rate: '折现率',
  start_state: '起始状态',
  credible_interval: '可信区间',
  prior_scale: '先验尺度',
  draws: '后验抽样次数',
  burn_in: '预热样本数',
  include_carryover: '纳入携带效应',
  model_type: '合并模型',
  tau_method: '异质性估计方法',
  effect_measure: '效应量类型',
  prediction_interval: '预测区间',
  ci_level: '置信水平',
  random_slope: '随机斜率',
  reml: 'REML 估计',
  optimizer: '优化器',
  weight_trim: '权重截尾分位',
  variance_method: '方差估计方法',
  domain_adjust: '亚组域校正',
  rg_threshold: '遗传相关阈值',
  threshold: '分类阈值',
  C: '惩罚参数 C',
  n_neighbors: '近邻数量 K',
  weights: '近邻权重',
  max_depth: '最大树深度',
  learning_rate: '学习率',
  kernel: '核函数',
  hidden_units: '隐藏单元结构',
  outlier_z: '异常值 Z 阈值',
  scaling: '标准化方式',
  encode_strategy: '分类变量编码',
  cv_folds: '交叉验证折数',
  class_weight: '类别权重',
  distance_metric: '距离度量',
  subsample: '样本采样比例',
  colsample_bytree: '特征采样比例',
  gamma: 'Gamma 参数',
  probability: '输出概率',
  criterion: '分裂准则',
  epochs: '训练轮数',
  batch_size: '批量大小',
  method: '分析方法',
  algorithm: '算法',
  n_clusters: '簇数',
  n_init: 'K-Means 初始化次数',
  n_components: '降维维度',
  perplexity: 't-SNE 困惑度',
};

const PARAM_EN_LABELS = {
  split_ratio: 'Train ratio',
  random_state: 'Random seed',
  cov_struct: 'Working correlation structure',
  max_iter: 'Maximum iterations',
  scale: 'Scale parameter',
  caliper: 'Matching caliper',
  matching_ratio: 'Matching ratio',
  ps_model: 'Propensity score model',
  standardize: 'Covariate standardization',
  missing_strategy: 'Missing-data strategy',
  alpha: 'Alpha regularization strength',
  alpha_mode: 'Alpha selection mode',
  bootstrap: 'Bootstrap resamples',
  delta_range: 'Perturbation range',
  effect_model: 'Treatment effect model',
  n_estimators: 'Number of estimators',
  min_samples_leaf: 'Minimum samples per leaf',
  trim_quantile: 'Extreme-value trimming quantile',
  time_horizon: 'Prediction time horizon',
  ties_method: 'Tie handling method',
  reference_group: 'Reference group',
  n_cycles: 'Number of cycles',
  cycle_length: 'Cycle length',
  discount_rate: 'Discount rate',
  start_state: 'Starting state',
  credible_interval: 'Credible interval',
  prior_scale: 'Prior scale',
  draws: 'Posterior draws',
  burn_in: 'Burn-in samples',
  include_carryover: 'Include carryover effect',
  model_type: 'Pooling model',
  tau_method: 'Tau-squared estimator',
  effect_measure: 'Effect measure',
  prediction_interval: 'Prediction interval',
  ci_level: 'Confidence level',
  random_slope: 'Random slope',
  reml: 'Restricted maximum likelihood',
  optimizer: 'Optimizer',
  weight_trim: 'Weight trimming quantile',
  variance_method: 'Variance estimation method',
  domain_adjust: 'Domain adjustment',
  rg_threshold: 'Genetic correlation threshold',
  threshold: 'Classification threshold',
  C: 'Penalty parameter C',
  n_neighbors: 'Number of neighbors K',
  weights: 'Neighbor weighting',
  max_depth: 'Maximum tree depth',
  learning_rate: 'Learning rate',
  kernel: 'Kernel function',
  hidden_units: 'Hidden-layer units',
  outlier_z: 'Outlier Z-score cutoff',
  scaling: 'Scaling method',
  encode_strategy: 'Categorical encoding',
  cv_folds: 'Cross-validation folds',
  class_weight: 'Class weighting',
  distance_metric: 'Distance metric',
  subsample: 'Row subsampling ratio',
  colsample_bytree: 'Column subsampling ratio',
  gamma: 'Gamma parameter',
  probability: 'Probability output',
  criterion: 'Split criterion',
  epochs: 'Training epochs',
  batch_size: 'Batch size',
  method: 'Method',
  algorithm: 'Algorithm',
  n_clusters: 'Number of clusters',
  n_init: 'K-Means initializations',
  n_components: 'Number of components',
  perplexity: 't-SNE perplexity',
};

const PARAM_OPTION_LABELS = {
  yes: '是 / yes',
  no: '否 / no',
  none: '无 / none',
  auto: '自动 / auto',
  balanced: '均衡 / balanced',
  uniform: '等权 / uniform',
  distance: '按距离加权 / distance',
  gaussian: '高斯 / Gaussian',
  binomial: '二项 / Binomial',
  poisson: '泊松 / Poisson',
  exchangeable: '可交换 / exchangeable',
  independence: '独立 / independence',
  ar1: '自回归 AR(1) / AR(1)',
  complete_case: '完整病例 / complete case',
  mean_impute: '均值填补 / mean impute',
  median_impute: '中位数填补 / median impute',
  median: '中位数 / median',
  mean: '均值 / mean',
  drop: '删除缺失行 / drop',
  zscore: 'Z 分数标准化 / z-score',
  minmax: '最小-最大缩放 / min-max',
  logistic: '逻辑回归 / logistic',
  random_forest: '随机森林 / random forest',
  t_learner: 'T-learner / T-learner',
  s_learner: 'S-learner / S-learner',
  random: '随机效应 / random effects',
  fixed: '固定效应 / fixed effects',
  linearized: '线性化 / linearized',
  replicate: '重复权重 / replicate weights',
  rbf: '径向基核 / RBF',
  linear: '线性核 / linear',
  poly: '多项式核 / polynomial',
  sigmoid: 'Sigmoid 核 / sigmoid',
  gini: 'Gini 指数 / Gini',
  entropy: '信息熵 / entropy',
  log_loss: '对数损失 / log loss',
  pca: '主成分分析 / PCA',
  tsne: 't-SNE / t-SNE',
  pca_tsne: 'PCA + t-SNE / PCA + t-SNE',
  kmeans: 'K-Means / K-Means',
  hierarchical: '层次聚类 / hierarchical',
  gmm: '高斯混合模型 / GMM',
  onehot: '独热编码 / one-hot',
  ordinal: '序数编码 / ordinal',
  minkowski: 'Minkowski 距离 / Minkowski',
  euclidean: '欧氏距离 / Euclidean',
  manhattan: '曼哈顿距离 / Manhattan',
};

function formula(title, math, note) {
  return { title, math: `<math class="formula-math" display="block" xmlns="http://www.w3.org/1998/Math/MathML">${math}</math>`, note };
}

const METHOD_FORMULAS = {
  gee: formula('GEE 均值模型与工作相关结构', '<mrow><mi>g</mi><mo>(</mo><mi>E</mi><mo>[</mo><msub><mi>Y</mi><mrow><mi>i</mi><mi>t</mi></mrow></msub><mo>]</mo><mo>)</mo><mo>=</mo><msubsup><mi>X</mi><mrow><mi>i</mi><mi>t</mi></mrow><mi>T</mi></msubsup><mi>β</mi><mo>,</mo><mspace width="0.5em"/><mi>Corr</mi><mo>(</mo><msub><mi>Y</mi><mrow><mi>i</mi><mi>t</mi></mrow></msub><mo>,</mo><msub><mi>Y</mi><mrow><mi>i</mi><mi>s</mi></mrow></msub><mo>)</mo><mo>=</mo><mi>R</mi><mo>(</mo><mi>α</mi><mo>)</mo></mrow>', '用连接函数刻画总体均值，同时用工作相关矩阵处理同一受试者内的相关性。'),
  propensity_score: formula('倾向评分与处理效应', '<mrow><mi>e</mi><mo>(</mo><mi>X</mi><mo>)</mo><mo>=</mo><mi>Pr</mi><mo>(</mo><mi>T</mi><mo>=</mo><mn>1</mn><mo>|</mo><mi>X</mi><mo>)</mo><mo>,</mo><mspace width="0.5em"/><mi>ATE</mi><mo>=</mo><mi>E</mi><mo>[</mo><mi>Y</mi><mo>(</mo><mn>1</mn><mo>)</mo><mo>-</mo><mi>Y</mi><mo>(</mo><mn>0</mn><mo>)</mo><mo>]</mo></mrow>', '先估计接受处理的概率，再在相近概率的人群之间比较结局。'),
  sensitivity_analysis: formula('扰动下的效应函数', '<mrow><mi>θ</mi><mo>(</mo><mi>δ</mi><mo>)</mo><mo>=</mo><mi>f</mi><mo>(</mo><mi>Y</mi><mo>,</mo><mi>T</mi><mo>,</mo><mi>X</mi><mo>;</mo><mi>δ</mi><mo>)</mo><mo>,</mo><mspace width="0.5em"/><mfrac><mrow><mi>∂</mi><mi>θ</mi></mrow><mrow><mi>∂</mi><mi>δ</mi></mrow></mfrac></mrow>', '观察关键结论在扰动参数变化时是否保持方向和强度。'),
  counterfactual: formula('反事实平均处理效应', '<mrow><mi>ATE</mi><mo>=</mo><mi>E</mi><mo>[</mo><mi>Y</mi><mo>(</mo><mn>1</mn><mo>)</mo><mo>-</mo><mi>Y</mi><mo>(</mo><mn>0</mn><mo>)</mo><mo>]</mo><mo>,</mo><mspace width="0.5em"/><mi>ATT</mi><mo>=</mo><mi>E</mi><mo>[</mo><mi>Y</mi><mo>(</mo><mn>1</mn><mo>)</mo><mo>-</mo><mi>Y</mi><mo>(</mo><mn>0</mn><mo>)</mo><mo>|</mo><mi>T</mi><mo>=</mo><mn>1</mn><mo>]</mo></mrow>', '把真实处理结果与未发生但可估计的反事实结果进行比较。'),
  survival_advanced: formula('Cox 风险函数与生存函数', '<mrow><mi>h</mi><mo>(</mo><mi>t</mi><mo>|</mo><mi>X</mi><mo>)</mo><mo>=</mo><msub><mi>h</mi><mn>0</mn></msub><mo>(</mo><mi>t</mi><mo>)</mo><mi>exp</mi><mo>(</mo><msup><mi>X</mi><mi>T</mi></msup><mi>β</mi><mo>)</mo><mo>,</mo><mspace width="0.5em"/><mi>S</mi><mo>(</mo><mi>t</mi><mo>)</mo><mo>=</mo><mi>exp</mi><mo>(</mo><mo>-</mo><mi>H</mi><mo>(</mo><mi>t</mi><mo>)</mo><mo>)</mo></mrow>', '描述事件发生风险随时间和协变量变化的规律。'),
  markov_model: formula('状态转移概率', '<mrow><msub><mi>π</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo><msub><mi>π</mi><mi>t</mi></msub><mi>P</mi><mo>,</mo><mspace width="0.5em"/><msub><mi>p</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><mi>Pr</mi><mo>(</mo><msub><mi>X</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo><mi>j</mi><mo>|</mo><msub><mi>X</mi><mi>t</mi></msub><mo>=</mo><mi>i</mi><mo>)</mo></mrow>', '用转移矩阵描述患者在不同疾病状态之间移动的概率。'),
  bayesian: formula('贝叶斯后验分布', '<mrow><mi>p</mi><mo>(</mo><mi>θ</mi><mo>|</mo><mi>y</mi><mo>)</mo><mo>=</mo><mfrac><mrow><mi>p</mi><mo>(</mo><mi>y</mi><mo>|</mo><mi>θ</mi><mo>)</mo><mi>p</mi><mo>(</mo><mi>θ</mi><mo>)</mo></mrow><mrow><mi>p</mi><mo>(</mo><mi>y</mi><mo>)</mo></mrow></mfrac></mrow>', '把先验知识和当前数据合成为后验分布。'),
  latin_square: formula('拉丁方方差分解', '<mrow><msub><mi>Y</mi><mrow><mi>i</mi><mi>j</mi><mi>k</mi></mrow></msub><mo>=</mo><mi>μ</mi><mo>+</mo><msub><mi>R</mi><mi>i</mi></msub><mo>+</mo><msub><mi>C</mi><mi>j</mi></msub><mo>+</mo><msub><mi>T</mi><mi>k</mi></msub><mo>+</mo><msub><mi>ε</mi><mrow><mi>i</mi><mi>j</mi><mi>k</mi></mrow></msub></mrow>', '把行、列和处理因素共同纳入，控制交叉设计中的系统差异。'),
  meta_analysis: formula('加权合并效应量', '<mrow><mover><mi>θ</mi><mo>^</mo></mover><mo>=</mo><mfrac><mrow><munderover><mo>Σ</mo><mi>i</mi><mi>k</mi></munderover><msub><mi>w</mi><mi>i</mi></msub><msub><mi>θ</mi><mi>i</mi></msub></mrow><mrow><munderover><mo>Σ</mo><mi>i</mi><mi>k</mi></munderover><msub><mi>w</mi><mi>i</mi></msub></mrow></mfrac><mo>,</mo><mspace width="0.5em"/><msub><mi>w</mi><mi>i</mi></msub><mo>=</mo><mfrac><mn>1</mn><mrow><msubsup><mi>SE</mi><mi>i</mi><mn>2</mn></msubsup><mo>+</mo><msup><mi>τ</mi><mn>2</mn></msup></mrow></mfrac></mrow>', '研究精度越高权重越大；随机效应模型额外考虑研究间异质性。'),
  mediation: formula('中介效应分解', '<mrow><mi>Total</mi><mo>=</mo><mi>Direct</mi><mo>+</mo><mi>Indirect</mi><mo>,</mo><mspace width="0.5em"/><mi>Indirect</mi><mo>=</mo><mi>a</mi><mo>×</mo><mi>b</mi></mrow>', '把自变量对结局的作用分成直接路径和通过中介变量的间接路径。'),
  mixed_effects: formula('混合效应模型', '<mrow><mi>Y</mi><mo>=</mo><mi>X</mi><mi>β</mi><mo>+</mo><mi>Z</mi><mi>b</mi><mo>+</mo><mi>ε</mi><mo>,</mo><mspace width="0.5em"/><mi>b</mi><mo>~</mo><mi>N</mi><mo>(</mo><mn>0</mn><mo>,</mo><mi>G</mi><mo>)</mo></mrow>', '固定效应解释总体趋势，随机效应吸收中心、个体等层级差异。'),
  nhanes_analysis: formula('复杂抽样加权均值', '<mrow><mover><mi>μ</mi><mo>^</mo></mover><mo>=</mo><mfrac><mrow><munderover><mo>Σ</mo><mi>i</mi><mi>n</mi></munderover><msub><mi>w</mi><mi>i</mi></msub><msub><mi>y</mi><mi>i</mi></msub></mrow><mrow><munderover><mo>Σ</mo><mi>i</mi><mi>n</mi></munderover><msub><mi>w</mi><mi>i</mi></msub></mrow></mfrac><mo>,</mo><mspace width="0.5em"/><mi>Var</mi><mo>(</mo><mover><mi>μ</mi><mo>^</mo></mover><mo>)</mo><mo>←</mo><mi>strata</mi><mo>,</mo><mi>PSU</mi></mrow>', '用抽样权重、分层和 PSU 恢复目标总体层面的估计。'),
  ldsc: formula('LDSC 遗传力与遗传相关', '<mrow><mi>E</mi><mo>[</mo><msubsup><mi>χ</mi><mi>j</mi><mn>2</mn></msubsup><mo>]</mo><mo>=</mo><mn>1</mn><mo>+</mo><mfrac><mrow><mi>N</mi><msup><mi>h</mi><mn>2</mn></msup></mrow><mi>M</mi></mfrac><msub><mi>l</mi><mi>j</mi></msub><mo>,</mo><mspace width="0.5em"/><msub><mi>r</mi><mi>g</mi></msub><mo>=</mo><mfrac><msub><mi>cov</mi><mi>g</mi></msub><msqrt><mrow><msubsup><mi>h</mi><mn>1</mn><mn>2</mn></msubsup><msubsup><mi>h</mi><mn>2</mn><mn>2</mn></msubsup></mrow></msqrt></mfrac></mrow>', '用 LD score 解释卡方统计量膨胀，并估计性状间共享遗传基础。'),
  ml_lr: formula('逻辑回归概率模型', '<mrow><mi>logit</mi><mo>(</mo><mi>p</mi><mo>)</mo><mo>=</mo><mi>log</mi><mfrac><mi>p</mi><mrow><mn>1</mn><mo>-</mo><mi>p</mi></mrow></mfrac><mo>=</mo><msup><mi>X</mi><mi>T</mi></msup><mi>β</mi></mrow>', '把线性预测值映射为 0 到 1 之间的疾病风险概率。'),
  ml_lasso: formula('Lasso L1 正则化', '<mrow><munder><mi>min</mi><mi>β</mi></munder><mspace width="0.4em"/><mo>||</mo><mi>y</mi><mo>-</mo><mi>X</mi><mi>β</mi><msubsup><mo>||</mo><mn>2</mn><mn>2</mn></msubsup><mo>+</mo><mi>λ</mi><munderover><mo>Σ</mo><mi>j</mi><mi>p</mi></munderover><mo>|</mo><msub><mi>β</mi><mi>j</mi></msub><mo>|</mo></mrow>', 'L1 惩罚会把部分系数压到 0，适合高维变量筛选。'),
  ml_ridge: formula('岭回归 L2 正则化', '<mrow><munder><mi>min</mi><mi>β</mi></munder><mspace width="0.4em"/><mo>||</mo><mi>y</mi><mo>-</mo><mi>X</mi><mi>β</mi><msubsup><mo>||</mo><mn>2</mn><mn>2</mn></msubsup><mo>+</mo><mi>λ</mi><munderover><mo>Σ</mo><mi>j</mi><mi>p</mi></munderover><msubsup><mi>β</mi><mi>j</mi><mn>2</mn></msubsup></mrow>', 'L2 惩罚收缩系数但通常不置零，适合缓解多重共线性。'),
  ml_knn: formula('K 近邻投票', '<mrow><mover><mi>y</mi><mo>^</mo></mover><mo>=</mo><mi>mode</mi><mo>{</mo><msub><mi>y</mi><mi>i</mi></msub><mo>:</mo><msub><mi>x</mi><mi>i</mi></msub><mo>∈</mo><msub><mi>N</mi><mi>k</mi></msub><mo>(</mo><mi>x</mi><mo>)</mo><mo>}</mo></mrow>', '根据距离最近的 K 个样本投票或加权投票得到预测。'),
  ml_xgboost: formula('梯度提升加法模型', '<mrow><msub><mover><mi>y</mi><mo>^</mo></mover><mi>i</mi></msub><mo>=</mo><munderover><mo>Σ</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mi>K</mi></munderover><msub><mi>f</mi><mi>k</mi></msub><mo>(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>)</mo><mo>,</mo><mspace width="0.5em"/><mi>Obj</mi><mo>=</mo><munderover><mo>Σ</mo><mi>i</mi><mi>n</mi></munderover><mi>L</mi><mo>(</mo><msub><mi>y</mi><mi>i</mi></msub><mo>,</mo><msub><mover><mi>y</mi><mo>^</mo></mover><mi>i</mi></msub><mo>)</mo><mo>+</mo><munderover><mo>Σ</mo><mi>k</mi><mi>K</mi></munderover><mi>Ω</mi><mo>(</mo><msub><mi>f</mi><mi>k</mi></msub><mo>)</mo></mrow>', '逐轮加入树模型，沿着损失函数下降方向提升预测表现。'),
  ml_rf: formula('随机森林集成预测', '<mrow><mover><mi>y</mi><mo>^</mo></mover><mo>=</mo><mfrac><mn>1</mn><mi>B</mi></mfrac><munderover><mo>Σ</mo><mrow><mi>b</mi><mo>=</mo><mn>1</mn></mrow><mi>B</mi></munderover><msub><mi>T</mi><mi>b</mi></msub><mo>(</mo><mi>x</mi><mo>)</mo></mrow>', '多棵随机树共同投票或平均，降低单棵树的不稳定性。'),
  ml_svm: formula('支持向量机最大间隔', '<mrow><munder><mi>min</mi><mrow><mi>w</mi><mo>,</mo><mi>b</mi><mo>,</mo><mi>ξ</mi></mrow></munder><mspace width="0.4em"/><mfrac><mn>1</mn><mn>2</mn></mfrac><mo>||</mo><mi>w</mi><msup><mo>||</mo><mn>2</mn></msup><mo>+</mo><mi>C</mi><munderover><mo>Σ</mo><mi>i</mi><mi>n</mi></munderover><msub><mi>ξ</mi><mi>i</mi></msub></mrow>', '在允许少量错分的条件下寻找最大分类间隔。'),
  ml_dt: formula('决策树分裂准则', '<mrow><msup><mi>s</mi><mo>*</mo></msup><mo>=</mo><munder><mi>argmax</mi><mi>s</mi></munder><mspace width="0.4em"/><mi>Δ</mi><mi>I</mi><mo>(</mo><mi>s</mi><mo>)</mo></mrow>', '每一步选择信息增益最大或不纯度下降最大的切分点。'),
  ml_cnn: formula('一维卷积特征提取', '<mrow><msub><mi>z</mi><mi>t</mi></msub><mo>=</mo><mi>σ</mi><mo>(</mo><msub><mrow><mo>(</mo><mi>W</mi><mo>*</mo><mi>X</mi><mo>)</mo></mrow><mi>t</mi></msub><mo>+</mo><mi>b</mi><mo>)</mo></mrow>', '卷积核沿时间窗口滑动，提取局部时序模式。'),
  feature_engineering: formula('特征转换流水线', '<mrow><msup><mi>X</mi><mo>*</mo></msup><mo>=</mo><msub><mi>φ</mi><mi>encode</mi></msub><mo>(</mo><msub><mi>φ</mi><mi>scale</mi></msub><mo>(</mo><msub><mi>φ</mi><mi>impute</mi></msub><mo>(</mo><mi>X</mi><mo>)</mo><mo>)</mo><mo>)</mo></mrow>', '把原始字段经过缺失处理、标准化、编码和衍生后变成可建模特征。'),
  model_comparison: formula('模型选择准则', '<mrow><msup><mi>m</mi><mo>*</mo></msup><mo>=</mo><munder><mi>argmax</mi><mi>m</mi></munder><mspace width="0.4em"/><mi>Score</mi><mo>(</mo><msub><mover><mi>y</mi><mo>^</mo></mover><mi>m</mi></msub><mo>,</mo><mi>y</mi><mo>)</mo></mrow>', '在同一数据划分和指标下比较多个候选模型。'),
  dim_reduction: formula('低维嵌入', '<mrow><mi>Z</mi><mo>=</mo><mi>X</mi><msub><mi>W</mi><mi>k</mi></msub><mo>,</mo><mspace width="0.5em"/><mi>dim</mi><mo>(</mo><mi>Z</mi><mo>)</mo><mo>≪</mo><mi>dim</mi><mo>(</mo><mi>X</mi><mo>)</mo></mrow>', '把高维矩阵压缩为低维坐标，保留主要方差或邻域结构。'),
  cluster: formula('聚类目标函数', '<mrow><munder><mi>min</mi><mrow><mi>C</mi><mo>,</mo><mi>μ</mi></mrow></munder><mspace width="0.4em"/><munderover><mo>Σ</mo><mi>i</mi><mi>n</mi></munderover><mo>||</mo><msub><mi>x</mi><mi>i</mi></msub><mo>-</mo><msub><mi>μ</mi><mrow><mi>c</mi><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msub><msup><mo>||</mo><mn>2</mn></msup></mrow>', '让同簇样本更接近、不同簇样本更分离。'),
};

const METHOD_CONCEPTS = {
  gee: ['重复测量数据', '工作相关结构', '稳健估计'],
  propensity_score: ['协变量建模', '倾向评分匹配', '处理效应'],
  sensitivity_analysis: ['基准模型', '扰动参数', '结论稳健性'],
  counterfactual: ['处理/对照', '反事实结局', '因果效应'],
  survival_advanced: ['随访时间', '风险函数', '生存曲线'],
  markov_model: ['疾病状态', '转移矩阵', '长期轨迹'],
  bayesian: ['先验信息', '似然数据', '后验推断'],
  latin_square: ['时期/序列', '处理因素', '方差分解'],
  meta_analysis: ['研究效应量', '权重合并', '异质性'],
  mediation: ['自变量 X', '中介 M', '结局 Y'],
  mixed_effects: ['固定效应', '随机效应', '个体轨迹'],
  nhanes_analysis: ['权重/分层/PSU', '复杂抽样设计', '总体推断'],
  ldsc: ['遗传力 h²', '标准误', '遗传相关 rg'],
  ml_lr: ['临床特征', '概率模型', '分类阈值'],
  ml_lasso: ['高维特征', 'L1 收缩', '变量筛选'],
  ml_ridge: ['共线特征', 'L2 收缩', '稳定预测'],
  ml_knn: ['标准化特征', '邻近样本', '投票分类'],
  ml_xgboost: ['弱学习器', '梯度提升', '特征贡献'],
  ml_rf: ['Bootstrap 样本', '多棵树', '集成预测'],
  ml_svm: ['核映射', '最大间隔', '支持向量'],
  ml_dt: ['分裂规则', '树结构', '临床决策'],
  ml_cnn: ['时序窗口', '卷积特征', '分类输出'],
  feature_engineering: ['原始字段', '清洗转换', '建模特征'],
  model_comparison: ['候选模型', '统一验证', '性能排序'],
  dim_reduction: ['高维矩阵', '低维嵌入', '结构可视化'],
  cluster: ['样本特征', '距离度量', '簇结构'],
};

const METHOD_CONCEPT_TYPES = {
  gee: 'longitudinal',
  propensity_score: 'matching',
  sensitivity_analysis: 'sensitivity',
  counterfactual: 'causal',
  survival_advanced: 'survival',
  markov_model: 'markov',
  bayesian: 'bayes',
  latin_square: 'latin',
  meta_analysis: 'forest',
  mediation: 'mediation',
  mixed_effects: 'longitudinal',
  nhanes_analysis: 'survey',
  ldsc: 'ldsc',
  ml_lr: 'sigmoid',
  ml_lasso: 'regularization',
  ml_ridge: 'regularization',
  ml_knn: 'knn',
  ml_xgboost: 'boosting',
  ml_rf: 'forest_model',
  ml_svm: 'svm',
  ml_dt: 'tree',
  ml_cnn: 'cnn',
  feature_engineering: 'pipeline',
  model_comparison: 'comparison',
  dim_reduction: 'embedding',
  cluster: 'clusters',
};

const METHOD_INTROS = {
  gee: '适合纵向、重复测量或同一受试者多时间点数据。它重点处理组内相关性，让总体平均效应估计更稳健。',
};

document.addEventListener('DOMContentLoaded', () => {
  bindNavigation();
  bindMethodNextAction();
  bindDataActions();
  bindRunAction();
  bindResultTabs();
  bindChartControls();
  renderAppearanceControls();
  loadMethodCatalog();
});

async function loadMethodCatalog() {
  try {
    const payload = await apiGet('/api/methods');
    APP.methods = (payload.methods || [])
      .map(method => ({ ...method, category: method.category === 'tools' ? 'ml_models' : method.category }))
      .filter(method => ['advanced_stats', 'ml_models'].includes(method.category));
    APP.methodMap = new Map(APP.methods.map(method => [method.id, method]));
    renderMethodCategories();
    renderMethodGrid();
    renderMethodDetail();
  } catch (error) {
    dom('miniMethodGrid').innerHTML = `<div class="empty-state error">方法目录载入失败：${html(error.message)}</div>`;
  }
}

function bindNavigation() {
  domAll('.nav-step').forEach(button => {
    button.addEventListener('click', () => setActiveTab(button.dataset.tab));
  });
}

function bindMethodNextAction() {
  const button = dom('methodNextBtn');
  if (!button) return;
  button.addEventListener('click', () => {
    const method = getActiveMethod();
    if (!method) return toast('请先选择统计方法', 'warning');
    setActiveTab('data');
  });
}

function setActiveTab(tab) {
  domAll('.nav-step').forEach(button => button.classList.toggle('active', button.dataset.tab === tab));
  domAll('.tab-panel').forEach(panel => panel.classList.toggle('active', panel.id === `tab-${tab}`));
  if (tab === 'result') {
    updateResultMethodSummary();
    setResultTab(APP.activeResultTab || 'chart');
    rerenderCurrentCharts();
  }
}

function renderMethodCategories() {
  domAll('.cat-tab', dom('methodCatTabs')).forEach(button => {
    button.addEventListener('click', () => {
      STATE.activeMethodCategory = button.dataset.cat;
      domAll('.cat-tab', dom('methodCatTabs')).forEach(item => item.classList.toggle('active', item === button));
      renderMethodGrid();
      renderMethodDetail();
    });
  });
}

function renderMethodGrid() {
  const category = STATE.activeMethodCategory || 'advanced_stats';
  const methods = APP.methods.filter(method => method.category === category);
  const grid = dom('miniMethodGrid');
  if (!methods.length) {
    grid.innerHTML = '<div class="empty-state">当前分类暂无方法。</div>';
    return;
  }

  grid.innerHTML = methods.map(method => `
    <button class="method-card ${STATE.activeMethodId === method.id ? 'active' : ''}" type="button" data-method-id="${html(method.id)}">
      <strong>${html(method.name)}</strong>
    </button>
  `).join('');

  domAll('[data-method-id]', grid).forEach(button => {
    button.addEventListener('click', () => selectMethod(button.dataset.methodId));
  });
}

function renderMethodDetail() {
  const container = dom('methodDetail');
  const nextButton = dom('methodNextBtn');
  if (!container) return;
  const method = getActiveMethod();
  if (nextButton) nextButton.disabled = !method;

  if (!method) {
    container.innerHTML = `
      <div class="method-detail-empty">
        <h1 id="method-title">请在左侧选择分析方法</h1>
      </div>
    `;
    return;
  }

  const intro = METHOD_INTROS[method.id] || method.description || `${method.name} 会根据所选变量和参数完成对应统计建模，并生成图表、表格和解释。`;
  const formulaDef = METHOD_FORMULAS[method.id] || formula('通用建模表达', '<mrow><mi>Model</mi><mo>=</mo><mi>f</mi><mo>(</mo><mi>X</mi><mo>,</mo><mi>Y</mi><mo>;</mo><mi>θ</mi><mo>)</mo></mrow>', '根据研究变量、协变量、结局变量和参数估计模型。');
  container.innerHTML = `
    <article class="method-detail-card">
      <p class="method-detail-kicker">当前统计方法</p>
      <h1 id="method-title">${html(method.name)}</h1>
      <section class="method-detail-block">
        <h2>基本概念</h2>
        <p>${html(intro)}</p>
      </section>
      <section class="method-detail-block">
        <h2>核心公式</h2>
        ${renderFormulaBox(formulaDef)}
      </section>
      <section class="method-detail-block">
        <h2>概念图</h2>
        ${renderConceptGraphic(method)}
      </section>
    </article>
  `;
}

function renderFormulaBox(item) {
  return `
    <div class="formula-box">
      <strong>${html(item.title || '核心公式')}</strong>
      <div class="formula-scroll">${item.math || html(String(item))}</div>
      ${item.note ? `<p>${html(item.note)}</p>` : ''}
    </div>
  `;
}

function renderConceptGraphic(method) {
  const labels = METHOD_CONCEPTS[method.id] || ['数据结构', '模型假设', '分析结果'];
  const type = METHOD_CONCEPT_TYPES[method.id] || 'pipeline';
  return `
    <div class="concept-map concept-map-${html(type)}" aria-label="${html(method.name)}概念图">
      ${renderConceptSvg(type, labels)}
      <div class="concept-caption-row">
        ${labels.map(label => `<span>${html(label)}</span>`).join('')}
      </div>
    </div>
  `;
}

function svgText(text, x, y, extra = '') {
  return `<text x="${x}" y="${y}" ${extra}>${html(text)}</text>`;
}

function renderConceptSvg(type, labels) {
  const l0 = labels[0] || '数据';
  const l1 = labels[1] || '模型';
  const l2 = labels[2] || '结果';
  const node = (x, y, label) => `
    <g class="svg-node">
      <rect x="${x - 36}" y="${y - 17}" width="72" height="34" rx="8"></rect>
      ${svgText(label, x, y + 4, 'text-anchor="middle"')}
    </g>
  `;
  const svgOpen = `<svg class="concept-svg" viewBox="0 0 260 168" role="img" aria-hidden="true">`;
  const svgClose = '</svg>';
  const defs = `
    <defs>
      <linearGradient id="conceptBlue" x1="0" x2="1" y1="0" y2="1">
        <stop offset="0%" stop-color="#f2f7ff"></stop>
        <stop offset="58%" stop-color="#dbeafe"></stop>
        <stop offset="100%" stop-color="#bfd4ff"></stop>
      </linearGradient>
      <linearGradient id="conceptMint" x1="0" x2="1" y1="0" y2="1">
        <stop offset="0%" stop-color="#ecfeff"></stop>
        <stop offset="100%" stop-color="#99f6e4"></stop>
      </linearGradient>
      <linearGradient id="conceptSlate" x1="0" x2="1" y1="0" y2="1">
        <stop offset="0%" stop-color="#f8fafc"></stop>
        <stop offset="100%" stop-color="#cbd5e1"></stop>
      </linearGradient>
      <linearGradient id="conceptLine" x1="0" x2="1" y1="0" y2="0">
        <stop offset="0%" stop-color="#2f6df6"></stop>
        <stop offset="100%" stop-color="#0ea5a4"></stop>
      </linearGradient>
      <marker id="arrowHead" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
        <path d="M0,0 L7,3.5 L0,7 Z" fill="#2563eb"></path>
      </marker>
    </defs>
  `;

  if (type === 'longitudinal') {
    return `${svgOpen}${defs}
      <path class="svg-axis" d="M32 132H232"></path>
      <path class="svg-axis" d="M42 136V28"></path>
      <path class="svg-curve one" d="M48 116 C82 88 112 104 148 70 S204 58 226 38"></path>
      <path class="svg-curve two" d="M48 124 C82 112 116 78 150 90 S204 70 226 62"></path>
      <path class="svg-curve three" d="M48 94 C82 86 110 66 148 58 S204 44 226 28"></path>
      ${[48,92,136,180,224].map(x => `<circle class="svg-dot" cx="${x}" cy="132" r="3"></circle>`).join('')}
      ${svgText('个体随访轨迹', 132, 153, 'text-anchor="middle" class="svg-label"')}
    ${svgClose}`;
  }
  if (type === 'matching') {
    return `${svgOpen}${defs}
      <g class="svg-cohort left">${[36,58,80,102].map((y, i) => `<circle cx="60" cy="${y}" r="9"></circle><circle cx="92" cy="${y + (i % 2 ? 5 : -5)}" r="9"></circle>`).join('')}</g>
      <g class="svg-cohort right">${[38,62,86,110].map((y, i) => `<circle cx="176" cy="${y}" r="9"></circle><circle cx="208" cy="${y + (i % 2 ? -4 : 4)}" r="9"></circle>`).join('')}</g>
      <path class="svg-arrow" d="M112 48 C132 36 150 36 168 48"></path>
      <path class="svg-arrow" d="M112 86 C132 78 150 78 168 86"></path>
      <path class="svg-arrow" d="M112 124 C132 116 150 116 168 124"></path>
      ${node(80, 148, l0)}${node(190, 148, l2)}
    ${svgClose}`;
  }
  if (type === 'sensitivity') {
    return `${svgOpen}${defs}
      <path class="svg-axis" d="M34 132H230"></path><path class="svg-axis" d="M44 138V30"></path>
      <path class="svg-band" d="M48 94 C90 70 132 78 176 58 S218 50 230 42 L230 80 C188 94 140 104 92 112 S54 126 48 124 Z"></path>
      <path class="svg-curve one" d="M48 110 C88 88 132 92 174 70 S216 62 230 54"></path>
      <line class="svg-threshold" x1="44" y1="92" x2="230" y2="92"></line>
      ${svgText('扰动参数 δ', 130, 153, 'text-anchor="middle" class="svg-label"')}
    ${svgClose}`;
  }
  if (type === 'causal') {
    return `${svgOpen}${defs}
      ${node(58, 66, 'T=1')}${node(58, 116, 'T=0')}${node(202, 66, 'Y(1)')}${node(202, 116, 'Y(0)')}
      <path class="svg-arrow" d="M96 66H164"></path><path class="svg-arrow" d="M96 116H164"></path>
      <path class="svg-dash" d="M58 83 C92 98 166 84 202 99"></path>
      ${svgText('反事实对照', 130, 28, 'text-anchor="middle" class="svg-label"')}
    ${svgClose}`;
  }
  if (type === 'survival') {
    return `${svgOpen}${defs}
      <path class="svg-axis" d="M34 132H232"></path><path class="svg-axis" d="M44 136V28"></path>
      <path class="svg-step one" d="M48 42H88V58H124V82H166V104H210V126"></path>
      <path class="svg-step two" d="M48 58H78V76H112V96H154V118H218V132"></path>
      ${svgText('S(t)', 28, 35, 'class="svg-label"')}${svgText('时间', 218, 154, 'class="svg-label"')}
    ${svgClose}`;
  }
  if (type === 'markov') {
    return `${svgOpen}${defs}
      ${['稳定', '进展', '缓解'].map((text, i) => {
        const points = [[60,72], [132,42], [200,104]][i];
        return `<g class="svg-state"><circle cx="${points[0]}" cy="${points[1]}" r="28"></circle>${svgText(text, points[0], points[1] + 5, 'text-anchor="middle"')}</g>`;
      }).join('')}
      <path class="svg-arrow" d="M88 62 C102 48 112 44 122 43"></path>
      <path class="svg-arrow" d="M153 53 C178 63 190 76 197 92"></path>
      <path class="svg-arrow" d="M176 116 C134 138 92 122 74 95"></path>
    ${svgClose}`;
  }
  if (type === 'bayes') {
    return `${svgOpen}${defs}
      ${node(58, 74, 'Prior')}${node(130, 38, 'Data')}${node(202, 74, 'Posterior')}
      <path class="svg-arrow" d="M93 64 C104 54 112 48 121 43"></path>
      <path class="svg-arrow" d="M139 43 C150 50 160 58 168 64"></path>
      <path class="svg-curve one" d="M44 126 C64 88 90 88 110 126"></path>
      <path class="svg-curve two" d="M150 126 C172 60 198 60 220 126"></path>
      ${svgText('先验 + 似然 → 后验', 132, 154, 'text-anchor="middle" class="svg-label"')}
    ${svgClose}`;
  }
  if (type === 'latin') {
    return `${svgOpen}${defs}
      <g class="svg-grid">${Array.from({ length: 16 }, (_, i) => {
        const x = 48 + (i % 4) * 42;
        const y = 26 + Math.floor(i / 4) * 28;
        const t = ['A', 'B', 'C', 'D'][(i + Math.floor(i / 4)) % 4];
        return `<rect x="${x}" y="${y}" width="34" height="22" rx="4"></rect>${svgText(t, x + 17, y + 15, 'text-anchor="middle"')}`;
      }).join('')}</g>
      ${svgText('行 / 列 / 处理平衡', 132, 153, 'text-anchor="middle" class="svg-label"')}
    ${svgClose}`;
  }
  if (type === 'forest') {
    return `${svgOpen}${defs}
      <line class="svg-threshold" x1="132" y1="24" x2="132" y2="132"></line>
      ${[42,66,90,114].map((y, i) => `<line class="svg-ci" x1="${64 + i * 10}" y1="${y}" x2="${174 - i * 6}" y2="${y}"></line><rect class="svg-square" x="${118 + i * 8}" y="${y - 6}" width="12" height="12" rx="2"></rect>`).join('')}
      <path class="svg-diamond" d="M108 140 L132 128 L156 140 L132 152 Z"></path>
      ${svgText('合并效应', 132, 22, 'text-anchor="middle" class="svg-label"')}
    ${svgClose}`;
  }
  if (type === 'mediation') {
    return `${svgOpen}${defs}
      ${node(50, 82, 'X')}${node(130, 42, 'M')}${node(210, 82, 'Y')}
      <path class="svg-arrow" d="M86 72 C102 52 112 44 121 42"></path>
      <path class="svg-arrow" d="M139 42 C154 44 168 54 176 72"></path>
      <path class="svg-arrow" d="M86 92H174"></path>
      ${svgText('间接路径 a × b', 132, 134, 'text-anchor="middle" class="svg-label"')}
    ${svgClose}`;
  }
  if (type === 'survey') {
    return `${svgOpen}${defs}
      <g class="svg-strata">${[40,92,144].map((x, i) => `<rect x="${x}" y="${34 + i * 12}" width="78" height="38" rx="8"></rect>${svgText(`Strata ${i + 1}`, x + 39, 58 + i * 12, 'text-anchor="middle"')}`).join('')}</g>
      <g>${[64,82,110,142,164,190].map((x, i) => `<circle class="svg-dot" cx="${x}" cy="${114 + (i % 2) * 14}" r="${5 + (i % 3)}"></circle>`).join('')}</g>
      ${svgText('权重 × 分层 × PSU', 132, 153, 'text-anchor="middle" class="svg-label"')}
    ${svgClose}`;
  }
  if (type === 'ldsc') {
    return `${svgOpen}${defs}
      ${node(62, 58, 'h² A')}${node(198, 58, 'h² B')}${node(130, 116, 'rg')}
      <path class="svg-arrow" d="M96 64 C118 86 122 96 126 100"></path>
      <path class="svg-arrow" d="M164 64 C142 86 138 96 134 100"></path>
      <path class="svg-curve two" d="M98 45 C128 24 166 24 190 45"></path>
      ${svgText('共享遗传基础', 132, 153, 'text-anchor="middle" class="svg-label"')}
    ${svgClose}`;
  }
  if (type === 'sigmoid') {
    return `${svgOpen}${defs}
      <path class="svg-axis" d="M34 132H232"></path><path class="svg-axis" d="M44 136V28"></path>
      <path class="svg-curve one" d="M42 124 C90 124 92 84 130 82 S172 40 226 40"></path>
      <line class="svg-threshold" x1="132" y1="132" x2="132" y2="40"></line>
      ${svgText('风险概率 p', 132, 153, 'text-anchor="middle" class="svg-label"')}
    ${svgClose}`;
  }
  if (type === 'regularization') {
    return `${svgOpen}${defs}
      <path class="svg-axis" d="M34 132H232"></path><path class="svg-axis" d="M44 136V28"></path>
      ${[54,78,102,126,150,174,198,222].map((x, i) => `<line class="svg-bar" x1="${x}" y1="132" x2="${x}" y2="${52 + (i % 4) * 13}"></line>`).join('')}
      <path class="svg-arrow" d="M54 28 C92 46 158 46 206 28"></path>
      ${svgText('系数收缩', 132, 153, 'text-anchor="middle" class="svg-label"')}
    ${svgClose}`;
  }
  if (type === 'knn') {
    return `${svgOpen}${defs}
      ${[54,74,88,104,66].map((x, i) => `<circle class="svg-point-a" cx="${x}" cy="${64 + i * 9}" r="6"></circle>`).join('')}
      ${[166,188,204,178,216].map((x, i) => `<circle class="svg-point-b" cx="${x}" cy="${48 + i * 14}" r="6"></circle>`).join('')}
      <circle class="svg-target" cx="126" cy="92" r="9"></circle>
      <circle class="svg-neighbor-ring" cx="126" cy="92" r="58"></circle>
      ${svgText('邻域投票', 132, 153, 'text-anchor="middle" class="svg-label"')}
    ${svgClose}`;
  }
  if (type === 'boosting') {
    return `${svgOpen}${defs}
      ${[42,88,134,180].map((x, i) => `<g class="svg-tree-mini"><path d="M${x + 18} 42V82"></path><path d="M${x + 18} 58H${x + 6}V78"></path><path d="M${x + 18} 58H${x + 30}V78"></path><circle cx="${x + 18}" cy="40" r="8"></circle><circle cx="${x + 6}" cy="82" r="6"></circle><circle cx="${x + 30}" cy="82" r="6"></circle></g>${i < 3 ? `<path class="svg-arrow" d="M${x + 44} 62H${x + 68}"></path>` : ''}`).join('')}
      ${svgText('逐轮提升', 132, 132, 'text-anchor="middle" class="svg-label"')}
    ${svgClose}`;
  }
  if (type === 'forest_model') {
    return `${svgOpen}${defs}
      ${[48,102,156].map(x => `<g class="svg-tree-mini"><path d="M${x + 22} 36V98"></path><path d="M${x + 22} 56H${x + 6}V92"></path><path d="M${x + 22} 56H${x + 38}V92"></path><circle cx="${x + 22}" cy="34" r="9"></circle><circle cx="${x + 6}" cy="96" r="7"></circle><circle cx="${x + 38}" cy="96" r="7"></circle></g>`).join('')}
      <path class="svg-arrow" d="M88 122H176"></path>${node(132, 142, 'Vote')}
    ${svgClose}`;
  }
  if (type === 'svm') {
    return `${svgOpen}${defs}
      <line class="svg-margin" x1="74" y1="26" x2="188" y2="138"></line>
      <line class="svg-margin dash" x1="52" y1="48" x2="166" y2="160"></line>
      <line class="svg-margin dash" x1="98" y1="4" x2="212" y2="116"></line>
      ${[56,74,88,96].map((x, i) => `<circle class="svg-point-a" cx="${x}" cy="${108 - i * 13}" r="6"></circle>`).join('')}
      ${[166,184,198,210].map((x, i) => `<circle class="svg-point-b" cx="${x}" cy="${42 + i * 15}" r="6"></circle>`).join('')}
      ${svgText('最大间隔', 132, 153, 'text-anchor="middle" class="svg-label"')}
    ${svgClose}`;
  }
  if (type === 'tree') {
    return `${svgOpen}${defs}
      ${node(130, 34, '规则')}${node(78, 88, '低风险')}${node(182, 88, '高风险')}${node(52, 136, 'A')}${node(104, 136, 'B')}${node(156, 136, 'C')}${node(208, 136, 'D')}
      <path class="svg-arrow" d="M116 48L92 72"></path><path class="svg-arrow" d="M144 48L168 72"></path>
      <path class="svg-arrow" d="M68 104L56 119"></path><path class="svg-arrow" d="M88 104L100 119"></path>
      <path class="svg-arrow" d="M172 104L160 119"></path><path class="svg-arrow" d="M192 104L204 119"></path>
    ${svgClose}`;
  }
  if (type === 'cnn') {
    return `${svgOpen}${defs}
      ${[34,56,78,100,122,144,166,188,210].map((x, i) => `<rect class="svg-signal" x="${x}" y="${86 - (i % 4) * 12}" width="12" height="${34 + (i % 4) * 12}" rx="3"></rect>`).join('')}
      <rect class="svg-window" x="72" y="32" width="62" height="92" rx="10"></rect>
      <path class="svg-arrow" d="M138 78H196"></path>
      ${svgText('卷积窗口', 103, 24, 'text-anchor="middle" class="svg-label"')}
    ${svgClose}`;
  }
  if (type === 'comparison') {
    return `${svgOpen}${defs}
      ${[54,92,130,168,206].map((x, i) => `<rect class="svg-bar-fill" x="${x}" y="${114 - i * 14}" width="20" height="${28 + i * 14}" rx="4"></rect>`).join('')}
      <path class="svg-axis" d="M34 132H232"></path><path class="svg-axis" d="M44 136V28"></path>
      ${svgText('统一指标比较', 132, 153, 'text-anchor="middle" class="svg-label"')}
    ${svgClose}`;
  }
  if (type === 'embedding') {
    return `${svgOpen}${defs}
      <g>${Array.from({ length: 30 }, (_, i) => `<circle class="${i % 3 === 0 ? 'svg-point-a' : i % 3 === 1 ? 'svg-point-b' : 'svg-point-c'}" cx="${48 + (i * 37) % 168}" cy="${36 + (i * 23) % 92}" r="5"></circle>`).join('')}</g>
      <path class="svg-arrow" d="M42 142H106"></path><path class="svg-arrow" d="M42 142V84"></path>
      ${svgText('二维嵌入', 132, 153, 'text-anchor="middle" class="svg-label"')}
    ${svgClose}`;
  }
  if (type === 'clusters') {
    return `${svgOpen}${defs}
      <ellipse class="svg-cluster" cx="72" cy="68" rx="40" ry="28"></ellipse>
      <ellipse class="svg-cluster two" cx="174" cy="72" rx="42" ry="30"></ellipse>
      <ellipse class="svg-cluster three" cx="128" cy="124" rx="44" ry="24"></ellipse>
      ${[54,70,84,96,66,182,160,196,150,132,118,142].map((x, i) => `<circle class="${i < 5 ? 'svg-point-a' : i < 9 ? 'svg-point-b' : 'svg-point-c'}" cx="${x}" cy="${i < 5 ? 58 + (i % 3) * 12 : i < 9 ? 60 + (i % 4) * 10 : 116 + (i % 2) * 12}" r="5"></circle>`).join('')}
    ${svgClose}`;
  }
  return `${svgOpen}${defs}
    ${node(52, 84, l0)}<path class="svg-arrow" d="M88 84H112"></path>${node(132, 84, l1)}<path class="svg-arrow" d="M168 84H192"></path>${node(212, 84, l2)}
  ${svgClose}`;
}

async function selectMethod(methodId) {
  const method = APP.methodMap.get(methodId);
  if (!method) return;
  const hadDemoData = Boolean(STATE.datasetName) && !STATE.uploadId;
  const previousMethodId = STATE.activeMethodId;

  STATE.activeMethodId = methodId;
  const variableMethodName = dom('variableMethodName');
  if (variableMethodName) variableMethodName.textContent = method.name;
  const dataMethodName = dom('dataMethodName');
  if (dataMethodName) dataMethodName.textContent = method.name;
  updateResultMethodSummary(method);
  const dataMethodHint = dom('dataMethodHint');
  if (dataMethodHint) dataMethodHint.textContent = `当前方法：${method.name}。请加载或上传数据后点击「下一步」进入变量选择界面。`;
  const selectedMethodLabel = dom('selectedMethodLabel');
  if (selectedMethodLabel) {
    selectedMethodLabel.textContent = `${method.name} · 示例数据：${method.example_dataset}.csv`;
  }

  if (hadDemoData) {
    resetDatasetState();
  } else if (previousMethodId !== methodId && STATE.columns.length) {
    clearMethodWorkspace(method.id);
  }

  clearResults();

  renderMethodGrid();
  renderMethodDetail();
  renderDataState();
  renderVariableControls();
  renderParamControls();
  if (STATE.columns.length) {
    recommendCurrentRoles({ quiet: true, render: true });
  }
}

function bindDataActions() {
  dom('loadExampleBtn').addEventListener('click', loadCurrentExample);
  const dataNext = dom('dataNextBtn');
  if (dataNext) {
    dataNext.addEventListener('click', () => {
      if (!STATE.columns.length) return toast('请先加载示例数据或上传数据', 'warning');
      setActiveTab('variables');
    });
  }
  dom('downloadExampleBtn').addEventListener('click', () => {
    const method = getActiveMethod();
    if (!method) return toast('请先选择统计方法', 'warning');
    apiDownload(`/api/examples/${method.example_dataset}/download`, `${method.example_dataset}.csv`);
  });
  dom('uploadDataBtn').addEventListener('click', () => dom('wsFileInput').click());
  dom('wsFileInput').addEventListener('change', event => {
    const file = event.target.files && event.target.files[0];
    if (file) uploadDataset(file);
    event.target.value = '';
  });
  dom('cancelUploadBtn').addEventListener('click', cancelUploadedDataset);
  renderUploadFileControl();
  renderDataNextControl();
}

function renderUploadFileControl() {
  const fileInput = dom('uploadedFileNameInput');
  const cancelBtn = dom('cancelUploadBtn');
  if (!fileInput || !cancelBtn) return;
  const hasUploadedFile = Boolean(STATE.uploadId);
  fileInput.value = hasUploadedFile ? (STATE.fileName || '') : '';
  fileInput.placeholder = hasUploadedFile ? '' : '未选择文件';
  cancelBtn.disabled = !hasUploadedFile;
}

function renderDataNextControl() {
  const button = dom('dataNextBtn');
  if (!button) return;
  button.disabled = !STATE.columns.length;
}

async function deleteUploadedFile(uploadId) {
  const response = await fetch(`/api/upload/${encodeURIComponent(uploadId)}`, { method: 'DELETE' });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || '取消上传失败');
  }
  return response.json();
}

async function cancelUploadedDataset() {
  const uploadId = STATE.uploadId;
  if (!uploadId) {
    renderUploadFileControl();
    return toast('当前没有用户上传的数据可取消', 'warning');
  }

  const button = dom('cancelUploadBtn');
  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = '取消中...';
  try {
    await deleteUploadedFile(uploadId);
    resetDatasetState();
    STATE.methodWorkspaces = {};
    APP.validationToken += 1;
    renderDataState();
    renderVariableControls();
    renderParamControls();
    clearResults();
    setActiveTab('data');
    toast('已取消上传的数据文件', 'success');
  } catch (error) {
    toast(error.message || '取消上传失败', 'error');
  } finally {
    button.textContent = originalText;
    renderUploadFileControl();
  }
}

async function loadCurrentExample() {
  const method = getActiveMethod();
  if (!method) return toast('请先选择统计方法', 'warning');
  const button = dom('loadExampleBtn');
  setLoading(button, true);
  try {
    const payload = await apiGet(`/api/examples/${method.example_dataset}`);
    applyDatasetPayload(payload, {
      uploadId: null,
      datasetName: method.example_dataset,
      fileName: payload.filename || `${method.example_dataset}.csv`,
    });
    renderDataState();
    renderVariableControls();
    renderParamControls();
    const recommendation = await recommendCurrentRoles({ quiet: true, render: true });
    renderDataNextControl();
    if (recommendation && recommendation.available) {
      setDataStatus('变量推荐完成，可以点击下一步。', 'ok');
    } else {
      setDataStatus('已加载示例数据，请检查变量要求', 'warning');
    }
  } catch (error) {
    toast(error.message || '示例数据载入失败', 'error');
  } finally {
    setLoading(button, false);
  }
}

async function uploadDataset(file) {
  const method = getActiveMethod();
  if (!method) {
    toast('请先选择统计方法，再上传数据', 'warning');
    return;
  }
  const button = dom('uploadDataBtn');
  setLoading(button, true);
  try {
    const form = new FormData();
    form.append('file', file);
    const response = await fetch('/api/upload', { method: 'POST', body: form });
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(err.detail || '上传失败');
    }
    const payload = await response.json();
    applyDatasetPayload(payload, {
      uploadId: payload.upload_id,
      datasetName: null,
      fileName: payload.filename || file.name,
    });
    renderDataState();
    renderVariableControls();
    renderParamControls();
    const recommendation = await recommendCurrentRoles({ quiet: true, render: true });
    renderDataNextControl();
    if (recommendation && recommendation.available) {
      setDataStatus('变量推荐完成，可以点击下一步。', 'ok');
    } else {
      setDataStatus('数据已上传，但当前统计方法可能无法运行', 'warning');
    }
  } catch (error) {
    toast(error.message || '上传失败', 'error');
  } finally {
    setLoading(button, false);
  }
}

function applyDatasetPayload(payload, meta) {
  STATE.uploadId = meta.uploadId || null;
  STATE.datasetName = meta.datasetName || null;
  STATE.fileName = meta.fileName || payload.filename || '';
  STATE.fileType = payload.file_type || '';
  STATE.sheetNames = payload.sheet_names || [];
  STATE.activeSheet = null;
  STATE.columns = payload.columns || [];
  STATE.dtypes = payload.dtypes || {};
  STATE.variableTypes = payload.variable_types || {};
  STATE.rowCount = payload.row_count || 0;
  STATE.colCount = payload.col_count || 0;
  STATE.previewRows = payload.preview || [];
  STATE.summary = payload.summary || {};
  clearResults();
  renderDataNextControl();
}

function renderDataState() {
  const method = getActiveMethod();
  const datasetLabel = STATE.columns.length
    ? (STATE.fileName || STATE.datasetName || (method ? `${method.example_dataset}.csv` : '已载入数据'))
    : '尚未载入数据';
  renderUploadFileControl();
  const activeDatasetName = dom('activeDatasetName');
  if (activeDatasetName) activeDatasetName.textContent = datasetLabel;
  const datasetMeta = dom('datasetMeta');
  if (datasetMeta) {
    datasetMeta.textContent = method
      ? `${method.name} 对应示例数据：${method.example_dataset}.csv`
      : '每个统计方法都有独立示例数据。';
  }

  if (!STATE.columns.length) {
    dom('wsDataMeta').textContent = method
      ? '可以加载该方法示例数据，也可以上传自己的 CSV / Excel 数据。'
      : '请先选择统计方法。';
    const dataMethodHint = dom('dataMethodHint');
    if (dataMethodHint) dataMethodHint.textContent = method
      ? `请加载或上传数据后点击「下一步」进入变量选择界面。`
      : '请先在方法选择界面选定分析方法，再加载或上传数据。';
    dom('previewTable').innerHTML = '<div class="empty-state small">等待数据载入</div>';
    renderDataNextControl();
    return;
  }

  dom('wsDataMeta').textContent = `${datasetLabel} · ${STATE.rowCount} 行 × ${STATE.colCount} 列`;
  dom('previewTable').innerHTML = renderPreviewTable();
  renderDataNextControl();

  const previewRows = (STATE.previewRows || []).length;
  const totalRows = STATE.rowCount || 0;
  const previewHint = dom('previewHint');
  if (previewHint) {
    if (previewRows > 0 && previewRows < totalRows) {
      previewHint.textContent = `⚠ 当前仅显示预览数据（前 ${previewRows} 行），完整数据（${totalRows} 行）已载入可用于分析。`;
    } else {
      previewHint.textContent = '';
    }
  }

  const dataMethodHint = dom('dataMethodHint');
  if (dataMethodHint && method) {
    dataMethodHint.textContent = `${method.name} · ${totalRows} 行 × ${STATE.colCount} 列。点击「下一步」进入变量选择界面。`;
  }
}

function renderPreviewTable() {
  const columns = (STATE.columns || []).slice(0, 12);
  const rows = (STATE.previewRows || []).slice(0, 30);
  if (!columns.length || !rows.length) return '<div class="empty-state small">暂无可预览数据。</div>';
  return `
    <table class="data-table">
      <thead><tr>${columns.map(col => `<th>${html(col)}</th>`).join('')}</tr></thead>
      <tbody>
        ${rows.map(row => `<tr>${columns.map(col => `<td>${html(row[col])}</td>`).join('')}</tr>`).join('')}
      </tbody>
    </table>
  `;
}

function renderVariableControls() {
  const container = dom('roleControls');
  if (!STATE.columns.length) {
    container.innerHTML = '<div class="empty-state small">请先载入数据。</div>';
    dom('roleSelectionSummary').textContent = '变量角色和方法参数会按当前统计方法变化。';
    showRoleValidation('载入示例数据或上传数据后，系统会按当前统计方法自动推荐变量。', 'muted');
    return;
  }

  const roles = getStoredRoles();
  const roleDefs = [
    ['research_vars', '研究变量 / 暴露 / 特征', '主要分组、暴露因素、模型特征'],
    ['covar_vars', '协变量 / 分层 / ID', '协变量、时间、ID、权重、标准误'],
    ['outcome_vars', '结局变量', '结局、目标变量、效应量、事件时间'],
  ];
  container.innerHTML = roleDefs.map(([key, label, hint]) => `
    <label class="role-box">
      <span>${label}</span>
      <small>${hint}</small>
      <select class="role-select" data-role="${key}" multiple size="9">
        ${(STATE.columns || []).map(col => `<option value="${html(col)}" ${roles[key].includes(col) ? 'selected' : ''}>${html(col)}${dtypeSuffix(col)}</option>`).join('')}
      </select>
    </label>
  `).join('');

  domAll('.role-select', container).forEach(select => {
    select.addEventListener('change', () => {
      storeRoles(collectRoles());
      updateRoleSummary();
      validateCurrentRoles({ quiet: false });
    });
  });
  dom('autoRoleBtn').onclick = () => {
    recommendCurrentRoles({ quiet: false, render: true });
  };
  updateRoleSummary();
}

function dtypeSuffix(column) {
  const dtype = STATE.dtypes && STATE.dtypes[column];
  return dtype ? ` · ${dtype}` : '';
}

function getStoredRoles() {
  const ws = getMethodWorkspace();
  return ws.roles || { research_vars: [], covar_vars: [], outcome_vars: [] };
}

function storeRoles(roles) {
  const method = getActiveMethod();
  if (!method) return;
  STATE.methodWorkspaces[method.id] = {
    ...(STATE.methodWorkspaces[method.id] || {}),
    roles,
  };
}

function clearMethodWorkspace(methodId) {
  if (!methodId) return;
  STATE.methodWorkspaces[methodId] = {
    ...(STATE.methodWorkspaces[methodId] || {}),
    roles: { research_vars: [], covar_vars: [], outcome_vars: [] },
    params: {},
  };
}

function collectRoles() {
  const roles = { research_vars: [], covar_vars: [], outcome_vars: [] };
  domAll('.role-select').forEach(select => {
    roles[select.dataset.role] = Array.from(select.selectedOptions).map(option => option.value);
  });
  return roles;
}

function updateRoleSummary() {
  const roles = collectRolesFromDomOrStore();
  const total = Object.values(roles).reduce((sum, values) => sum + values.length, 0);
  dom('roleSelectionSummary').textContent = total
    ? `已选择 ${total} 个变量，可继续调节下方参数。`
    : '请选择变量，或点击自动推荐。';
}

function collectRolesFromDomOrStore() {
  return domAll('.role-select').length ? collectRoles() : getStoredRoles();
}

function showRoleValidation(message, type = 'muted') {
  const box = dom('roleValidationMessage');
  if (!box) return;
  box.className = `validation-message ${type}`;
  box.textContent = message;
}

function setDataStatus(message, type = 'muted', durationMs = 5000) {
  const bar = dom('dataStatusBar');
  if (!bar) return;
  bar.className = `data-status-bar ${type}`;
  bar.textContent = message;
  if (durationMs > 0) {
    setTimeout(() => {
      if (bar.textContent === message) {
        bar.className = 'data-status-bar';
        bar.textContent = '';
      }
    }, durationMs);
  }
}

async function validateCurrentRoles(options = {}) {
  const method = getActiveMethod();
  if (!method || !STATE.columns.length) {
    showRoleValidation('载入示例数据或上传数据后，系统会按当前统计方法自动推荐变量。', 'muted');
    return null;
  }

  const roles = collectRolesFromDomOrStore();
  const roleCount = Object.values(roles).reduce((sum, values) => sum + (values || []).length, 0);
  if (!roleCount) {
    showRoleValidation('请先在左侧选择变量，或点击“自动推荐”。', 'muted');
    return { available: false, reason: '请先选择变量。' };
  }

  const token = ++APP.validationToken;
  if (!options.quiet) {
    showRoleValidation('正在校验这些变量是否适合当前方法...', 'muted');
  }

  try {
    const payload = await apiPost('/api/validate-methods', {
      upload_id: STATE.uploadId,
      dataset_name: STATE.uploadId ? null : (STATE.datasetName || method.example_dataset),
      use_demo: !STATE.uploadId,
      sheet_name: STATE.activeSheet,
      roles,
    });
    const item = payload.availability && payload.availability[method.id];
    if (token !== APP.validationToken) return item || null;
    if (item && item.available) {
      showRoleValidation('变量选择可用于当前方法。可以继续调节右侧参数并开始分析。', 'ok');
    } else {
      showRoleValidation(`当前变量不能运行「${method.name}」：${(item && item.reason) || '变量组合不满足该方法要求。'}`, 'error');
    }
    return item || null;
  } catch (error) {
    if (token === APP.validationToken) {
      showRoleValidation(`变量校验失败：${error.message || '请检查数据与变量选择。'}`, 'error');
    }
    return { available: false, reason: error.message || '变量校验失败。' };
  }
}

function defaultMethodParams(method) {
  const params = {};
  (method.params || []).forEach(param => {
    if (param.default !== undefined && param.default !== '') params[param.key] = param.default;
  });
  return params;
}

async function recommendCurrentRoles(options = {}) {
  const method = getActiveMethod();
  if (!method || !STATE.columns.length) {
    showRoleValidation('请先选择统计方法并载入数据。', 'muted');
    return null;
  }

  if (!options.quiet) {
    showRoleValidation('正在浏览整张数据并判断是否适合当前统计方法...', 'muted');
  }

  try {
    const payload = await apiPost('/api/recommend-roles', {
      method_id: method.id,
      upload_id: STATE.uploadId,
      dataset_name: STATE.uploadId ? null : (STATE.datasetName || method.example_dataset),
      use_demo: !STATE.uploadId,
      sheet_name: STATE.activeSheet,
    });

    if (payload.available) {
      const roles = payload.roles || { research_vars: [], covar_vars: [], outcome_vars: [] };
      STATE.methodWorkspaces[method.id] = {
        ...(STATE.methodWorkspaces[method.id] || {}),
        roles,
        params: { ...defaultMethodParams(method), ...(payload.params || {}) },
      };
      if (options.render) {
        renderVariableControls();
        renderParamControls();
      } else {
        updateRoleSummary();
      }
      showRoleValidation(`已浏览整张数据，当前文件可以运行「${method.name}」。已自动推荐一组可运行变量。`, 'ok');
    } else {
      clearMethodWorkspace(method.id);
      if (options.render) {
        renderVariableControls();
        renderParamControls();
      } else {
        updateRoleSummary();
      }
      showRoleValidation(`上传的数据文件做不了「${method.name}」：${payload.reason || '没有找到满足该统计方法要求的变量组合。'}`, 'error');
    }
    return payload;
  } catch (error) {
    showRoleValidation(`自动推荐失败：${error.message || '请检查数据文件和统计方法。'}`, 'error');
    return { available: false, reason: error.message || '自动推荐失败' };
  }
}

function renderParamControls() {
  const method = getActiveMethod();
  const container = dom('paramControls');
  if (!method) {
    dom('paramCountLabel').textContent = '0 项参数';
    container.innerHTML = '<div class="empty-state small">选择方法后显示该方法可调参数。</div>';
    return;
  }

  const params = buildMethodParamList(method);
  dom('paramCountLabel').textContent = `${params.length} 项参数`;
  if (!params.length) {
    container.innerHTML = '<div class="empty-state small">该方法的变量在左侧选择；当前没有额外参数需要调节。</div>';
    return;
  }

  const saved = getMethodWorkspace().params || {};
  container.innerHTML = params.map(param => renderParamField(param, saved[param.key])).join('');
  domAll('[data-param-key]', container).forEach(input => {
    input.addEventListener('change', () => saveCurrentParams());
    input.addEventListener('input', () => saveCurrentParams());
  });
}

function buildMethodParamList(method) {
  return [...(method.params || [])]
    .map(param => ({ ...param }))
    .filter(param => !isVariableParam(param));
}

function isVariableParam(param) {
  const key = String(param.key || '');
  const type = String(param.type || '');
  if (!key) return false;
  if (['target', 'features', 'feature_vars', 'covariates', 'covariate_vars'].includes(key)) return true;
  if (/(_var|_vars|_col|_cols)$/.test(key)) return true;
  if (type === 'multi_select') return true;
  if (type === 'select' && !(Array.isArray(param.options) && param.options.length)) return true;
  return false;
}

function renderParamLabel(param) {
  const key = String(param.key || '');
  const cn = PARAM_CN_LABELS[key] || param.label || key;
  const en = param.label_en || PARAM_EN_LABELS[key] || key;
  return `
    <span class="param-label">
      <strong>${html(cn)}</strong>
      <em>${html(en)}</em>
    </span>
  `;
}

function renderParamOption(option) {
  return PARAM_OPTION_LABELS[String(option)] || option;
}

function renderParamField(param, savedValue) {
  const key = param.key;
  const value = savedValue !== undefined ? savedValue : param.default;
  const type = param.type || 'text';

  if (type === 'select' && Array.isArray(param.options) && param.options.length) {
    return `
      <label class="field">
        ${renderParamLabel(param)}
        <select data-param-key="${html(key)}">
          ${param.options.map(option => `<option value="${html(option)}" ${String(option) === String(value) ? 'selected' : ''}>${html(renderParamOption(option))}</option>`).join('')}
        </select>
      </label>
    `;
  }

  if (type === 'select' || key.endsWith('_var') || key === 'target') {
    return `
      <label class="field">
        ${renderParamLabel(param)}
        <select data-param-key="${html(key)}">
          <option value="">不指定</option>
          ${(STATE.columns || []).map(col => `<option value="${html(col)}" ${String(col) === String(value) ? 'selected' : ''}>${html(col)}</option>`).join('')}
        </select>
      </label>
    `;
  }

  if (type === 'multi_select' || key.endsWith('_vars')) {
    const selected = Array.isArray(value) ? value : [];
    return `
      <label class="field wide">
        ${renderParamLabel(param)}
        <select data-param-key="${html(key)}" multiple size="8">
          ${(STATE.columns || []).map(col => `<option value="${html(col)}" ${selected.includes(col) ? 'selected' : ''}>${html(col)}${dtypeSuffix(col)}</option>`).join('')}
        </select>
      </label>
    `;
  }

  if (type === 'checkbox') {
    return `
      <label class="check-field">
        <input data-param-key="${html(key)}" type="checkbox" ${value ? 'checked' : ''} />
        ${renderParamLabel(param)}
      </label>
    `;
  }

  if (type === 'number') {
    return `
      <label class="field">
        ${renderParamLabel(param)}
        <input data-param-key="${html(key)}" type="number" value="${html(value ?? '')}" min="${html(param.min ?? '')}" max="${html(param.max ?? '')}" step="${html(param.step ?? 'any')}" />
      </label>
    `;
  }

  return `
    <label class="field">
      ${renderParamLabel(param)}
      <input data-param-key="${html(key)}" type="text" value="${html(value ?? '')}" />
    </label>
  `;
}

function collectParams() {
  const params = {};
  domAll('[data-param-key]', dom('paramControls')).forEach(input => {
    const key = input.dataset.paramKey;
    if (input.multiple) {
      params[key] = Array.from(input.selectedOptions).map(option => option.value);
    } else if (input.type === 'checkbox') {
      params[key] = input.checked;
    } else if (input.type === 'number') {
      params[key] = input.value === '' ? '' : Number(input.value);
    } else {
      params[key] = input.value;
    }
  });
  return params;
}

function saveCurrentParams() {
  const method = getActiveMethod();
  if (!method) return;
  STATE.methodWorkspaces[method.id] = {
    ...(STATE.methodWorkspaces[method.id] || {}),
    params: collectParams(),
  };
}

function autoFillRoles() {
  const method = getActiveMethod();
  if (!method || !STATE.columns.length) return;
  const roles = buildAutoRoles(method);
  storeRoles(roles);
  const ws = getMethodWorkspace();
  ws.params = { ...(ws.params || {}), ...buildAutoParams(method, roles) };
  STATE.methodWorkspaces[method.id] = ws;
}

function buildAutoRoles(method) {
  const cols = STATE.columns || [];
  const defaults = Object.fromEntries((method.params || []).map(param => [param.key, param.default]));
  const numeric = cols.filter(isNumericColumn).filter(col => !isIdLike(col));
  const categorical = cols.filter(col => !isNumericColumn(col) && !isIdLike(col));
  const pick = (...names) => names.find(name => name && cols.includes(name));
  const byName = (pattern, pool = cols, predicate = () => true) => {
    const regex = new RegExp(pattern, 'i');
    return (pool || []).find(col => regex.test(String(col)) && predicate(col));
  };
  const numericByName = (pattern, pool = cols) => byName(pattern, pool, isNumericColumn);
  const categoricalByName = (pattern, pool = cols) => byName(pattern, pool, col => !isNumericColumn(col) && !isIdLike(col));
  const previewUniqueCount = (column) => {
    const values = new Set((STATE.previewRows || [])
      .map(row => row && row[column])
      .filter(value => value !== undefined && value !== null && String(value) !== ''));
    return values.size;
  };
  const isLikelyBinaryColumn = (column) => {
    const count = previewUniqueCount(column);
    return count === 2 || (!isNumericColumn(column) && /sex|gender|binary|case|control/i.test(String(column)));
  };
  const binaryByName = (pattern, pool = cols) => byName(pattern, pool, col => !isIdLike(col) && isLikelyBinaryColumn(col));
  const addMany = (bucket, values) => values.forEach(value => add(bucket, value));
  const target = pick(defaults.target, defaults.outcome_var, defaults.response_var, defaults.y_var, defaults.effect_var, defaults.h2_col);
  const group = pick(defaults.group_var, defaults.treatment_var, defaults.state_var, defaults.formulation_var, defaults.study_var, defaults.x_var);
  const time = pick(defaults.time_var, defaults.period_var);
  const subject = pick(defaults.subject_var, defaults.random_var);
  const event = pick(defaults.event_var);
  const se = pick(defaults.se_var, defaults.h2_se_col);
  const baseline = pick(defaults.baseline_var, defaults.m_var, defaults.weight_var, defaults.strata_var);

  const research = [];
  const covars = [];
  const outcomes = [];
  const add = (bucket, value) => {
    if (value && cols.includes(value) && !bucket.includes(value)) bucket.push(value);
  };

  if (method.id === 'survival_advanced') {
    add(research, group);
    add(outcomes, time);
    add(outcomes, event);
  } else if (method.id === 'sensitivity_analysis') {
    const treatment = binaryByName('treat|treated|intervention|exposure') || binaryByName('sex|gender|smoking') || binaryByName('arm|group') || categoricalByName('sex|gender|smoking|arm|group');
    const outcome = pick(defaults.outcome_var) || numericByName('followup|outcome|score|response|change') || numeric[0];
    const baselineVar = pick(defaults.baseline_var) || numericByName('baseline|base|pre') || numeric.find(col => col !== outcome);
    add(research, treatment);
    add(covars, baselineVar);
    addMany(covars, numeric.filter(col => ![treatment, outcome, baselineVar].includes(col)).slice(0, 3));
    add(outcomes, outcome);
  } else if (method.id === 'markov_model') {
    add(research, pick(defaults.state_var) || categoricalByName('^state$|status|stage'));
    add(covars, pick(defaults.subject_var) || byName('subject|patient|person|sample|id', cols, col => /subject|patient|sample|id/i.test(String(col))));
    add(covars, time || numericByName('month|time|cycle|period|visit'));
  } else if (method.id === 'meta_analysis') {
    add(research, defaults.study_var);
    add(covars, se);
    add(outcomes, defaults.effect_var);
  } else if (method.id === 'nhanes_analysis') {
    add(covars, pick(defaults.weight_var) || numericByName('weight|wt|survey_weight'));
    add(covars, pick(defaults.strata_var) || categoricalByName('strata|stratum|site|center'));
    add(covars, byName('psu|cluster', cols, col => !isIdLike(col)));
    add(outcomes, pick(defaults.outcome_var) || numericByName('sbp|outcome|glucose|hba1c|bmi') || numeric[0]);
  } else if (method.id === 'ldsc') {
    add(research, pick(defaults.group_var) || categoricalByName('^trait$|phenotype|disease'));
    add(covars, pick(defaults.h2_se_col) || numericByName('h2.*se|se.*h2|standard.*error'));
    add(outcomes, pick(defaults.h2_col) || numericByName('^h2$|heritability'));
  } else if (method.id === 'ml_lasso' || method.id === 'ml_ridge') {
    const regressionTarget = pick(defaults.target)
      || (method.id === 'ml_ridge' ? numericByName('renal_metabolic_index') : null)
      || numericByName('^outcome$|outcome_continuous|response|target|followup|score|index')
      || numeric[numeric.length - 1];
    add(outcomes, regressionTarget);
    addMany(research, numeric.filter(col => col !== regressionTarget).slice(0, 12));
    addMany(covars, categorical.filter(col => col !== regressionTarget).slice(0, 2));
  } else if (method.category === 'ml_models') {
    add(outcomes, defaults.target);
    numeric.filter(col => col !== defaults.target).slice(0, 8).forEach(col => add(research, col));
    categorical.filter(col => col !== defaults.target).slice(0, 2).forEach(col => add(covars, col));
  } else {
    add(research, group);
    add(covars, time);
    add(covars, subject);
    add(covars, se);
    add(covars, baseline);
    add(outcomes, target);
  }

  if (method.id === 'propensity_score' || method.id === 'counterfactual') {
    numeric.filter(col => col !== target && col !== group).slice(0, 4).forEach(col => add(covars, col));
  }
  if (method.id === 'mediation') {
    add(research, defaults.x_var);
    add(covars, defaults.m_var);
    add(outcomes, defaults.y_var);
  }
  if (method.id === 'dim_reduction' || method.id === 'cluster' || method.id === 'feature_engineering') {
    numeric.slice(0, 8).forEach(col => add(research, col));
    add(covars, defaults.group_var);
  }

  return { research_vars: research, covar_vars: covars, outcome_vars: outcomes };
}

function buildAutoParams(method, roles) {
  const params = {};
  (method.params || []).forEach(param => {
    if (param.default !== undefined && param.default !== '') params[param.key] = param.default;
  });
  if (method.category === 'ml_models') {
    params.feature_vars = [...(roles.research_vars || []), ...(roles.covar_vars || [])].filter(col => col !== params.target);
  }
  if (method.id === 'cluster' || method.id === 'dim_reduction' || method.id === 'feature_engineering') {
    params.feature_vars = roles.research_vars || [];
  }
  return params;
}

function isNumericColumn(column) {
  const dtype = String((STATE.dtypes || {})[column] || '').toLowerCase();
  return /(int|float|double|number|decimal)/.test(dtype);
}

function isIdLike(column) {
  return /(^id$|_id$|id$|subject|patient|sample)/i.test(String(column));
}

function getActiveMethod() {
  return APP.methodMap.get(STATE.activeMethodId);
}

function updateResultMethodSummary(method = getActiveMethod()) {
  const resultMethodName = dom('resultMethodName');
  if (resultMethodName) {
    resultMethodName.textContent = method ? method.name : '尚未选择方法';
  }
}

function getMethodWorkspace() {
  const method = getActiveMethod();
  if (!method) return {};
  STATE.methodWorkspaces[method.id] = STATE.methodWorkspaces[method.id] || {};
  return STATE.methodWorkspaces[method.id];
}

function bindRunAction() {
  dom('generateBtn').addEventListener('click', runAnalysis);
}

async function runAnalysis() {
  const method = getActiveMethod();
  if (!method) return toast('请先选择统计方法', 'warning');
  if (!STATE.columns.length) return toast('请先上传数据或加载示例数据', 'warning');

  saveCurrentParams();
  storeRoles(collectRolesFromDomOrStore());
  const roles = collectRolesFromDomOrStore();
  const roleCount = Object.values(roles).reduce((sum, values) => sum + values.length, 0);
  if (!roleCount) {
    showRoleValidation('请先在左侧选择变量，或点击“自动推荐”。', 'error');
    setActiveTab('variables');
    return toast('请先选择适合当前方法的变量', 'warning');
  }

  const validation = await validateCurrentRoles({ quiet: false });
  if (!validation || !validation.available) {
    setActiveTab('variables');
    return toast((validation && validation.reason) || '当前变量不能运行该分析方法', 'error');
  }

  const params = collectParams();
  const payload = {
    method_id: method.id,
    upload_id: STATE.uploadId,
    sheet_name: STATE.activeSheet,
    use_demo: !STATE.uploadId,
    dataset_name: STATE.uploadId ? null : (STATE.datasetName || method.example_dataset),
    params,
    role_vars: roleCount ? roles : {},
  };

  const button = dom('generateBtn');
  setLoading(button, true);
  try {
    const result = await apiPost('/api/analyze', payload);
    APP.lastResult = result;
    APP.activeResultTab = 'chart';
    STATE.activeChartIndex = 0;
    renderResults(result);
    setActiveTab('result');
    setResultTab('chart');
    rerenderCurrentCharts();
    toast('分析完成', 'success');
  } catch (error) {
    showRoleValidation(`分析失败：${error.message || '请检查变量与参数设置。'}`, 'error');
    toast(error.message || '分析失败', 'error');
  } finally {
    setLoading(button, false);
  }
}

function clearResults() {
  APP.lastResult = null;
  updateResultMethodSummary();
  dom('resultSummary').textContent = '右侧是结果区域，运行分析后可在此查看可视化、统计结果和结果解读。';
  dom('chartPreviewContainer').innerHTML = '<div id="chartActivePlot" class="chart-active-plot"><div class="empty-state">完成分析后展示结果图形。</div></div>';
  dom('chartVariantTabs').innerHTML = '';
  dom('resultTablesContainer').innerHTML = '<div class="empty-state small">表格结果将在分析后显示。</div>';
  dom('discussionContainer').innerHTML = '<div class="empty-state small">结果解释将在分析后显示。</div>';
}

function renderResults(result) {
  updateResultMethodSummary();
  const datasetLabel = STATE.fileName || STATE.datasetName || '';
  dom('resultSummary').textContent = datasetLabel
    ? `右侧是结果区域，当前结果来自 ${datasetLabel}。可在此查看可视化、统计结果和结果解读。`
    : '右侧是结果区域，可在此查看可视化、统计结果和结果解读。';
  renderResultTables(result.tables || []);
  renderDiscussion(result.discussion || '');
  if (typeof renderAllCharts === 'function') {
    renderAllCharts(result.charts || []);
  } else {
    dom('chartPreviewContainer').innerHTML = '<div class="empty-state error">图表渲染模块不可用。</div>';
  }
  renderAppearanceControls();
}

function renderResultTables(tables) {
  const container = dom('resultTablesContainer');
  APP.lastResultTables = Array.isArray(tables) ? tables : [];

  const tablesHtml = tables.length ? tables.map((table, index) => {
    const rows = table.rows || table.data || [];
    const headers = table.headers || table.columns || inferHeaders(rows);
    return `
      <section class="table-block">
        <div class="table-block-head">
          <h2>${html(table.title || `结果表 ${index + 1}`)}</h2>
          <div class="table-actions" aria-label="表格导出">
            <button class="table-export-btn" data-table-export="csv" data-table-index="${index}" type="button">CSV</button>
            <button class="table-export-btn" data-table-export="txt" data-table-index="${index}" type="button">TXT</button>
            <button id="table-export-pdf-${index + 1}" class="table-export-btn" data-table-export="pdf" data-table-index="${index}" type="button">PDF</button>
          </div>
        </div>
        ${renderDataTable(headers, rows)}
      </section>
    `;
  }).join('') : '<div class="empty-state small">本次分析没有返回表格。</div>';

  container.innerHTML = tablesHtml;
  domAll('[data-table-export]', container).forEach(button => {
    button.addEventListener('click', () => exportResultTable(Number(button.dataset.tableIndex), button.dataset.tableExport));
  });
}

function inferHeaders(rows) {
  const first = rows && rows[0];
  if (!first) return [];
  return Array.isArray(first) ? first.map((_, index) => `列 ${index + 1}`) : Object.keys(first);
}

function renderDataTable(headers, rows) {
  if (!headers.length || !rows.length) return '<div class="empty-state small">暂无表格数据。</div>';
  const cells = row => headers.map(header => `<td>${html(Array.isArray(row) ? row[headers.indexOf(header)] : row[header])}</td>`).join('');
  return `
    <table class="data-table">
      <thead><tr>${headers.map(header => `<th>${html(header)}</th>`).join('')}</tr></thead>
      <tbody>${rows.map(row => `<tr>${cells(row)}</tr>`).join('')}</tbody>
    </table>
  `;
}

function tableRowsAndHeaders(table) {
  const rows = (table && (table.rows || table.data)) || [];
  const headers = (table && (table.headers || table.columns)) || inferHeaders(rows);
  return { headers, rows };
}

function tableCell(row, header, index) {
  if (Array.isArray(row)) return row[index] ?? '';
  if (row && typeof row === 'object') return row[header] ?? '';
  return '';
}

function rowsToCsv(headers, rows) {
  const matrix = [headers, ...rows.map(row => headers.map((header, index) => tableCell(row, header, index)))];
  return matrix.map(line => line.map(cell => `"${String(cell ?? '').replaceAll('"', '""')}"`).join(',')).join('\n');
}

function rowsToTxt(headers, rows) {
  const matrix = [headers, ...rows.map(row => headers.map((header, index) => tableCell(row, header, index)))];
  return matrix.map(line => line.map(cell => String(cell ?? '').replace(/\s+/g, ' ').trim()).join('\t')).join('\n');
}

function rowsToHtmlTable(headers, rows, title) {
  const head = headers.map(header => `<th>${html(header)}</th>`).join('');
  const body = rows.map(row => `<tr>${headers.map((header, index) => `<td>${html(tableCell(row, header, index))}</td>`).join('')}</tr>`).join('');
  return `<!doctype html>
    <html lang="zh-CN">
      <head>
        <meta charset="utf-8" />
        <title>${html(title)}</title>
        <style>
          body { font-family: Arial, "Microsoft YaHei", sans-serif; padding: 24px; color: #111827; }
          h1 { font-size: 18px; margin: 0 0 16px; }
          table { width: 100%; border-collapse: collapse; font-size: 12px; }
          th, td { border: 1px solid #d1d5db; padding: 6px 8px; text-align: left; }
          th { background: #f3f4f6; font-weight: 700; }
          @media print { body { padding: 0; } }
        </style>
      </head>
      <body>
        <h1>${html(title)}</h1>
        <table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>
      </body>
    </html>`;
}

function safeFilenamePart(value) {
  return String(value || 'table')
    .replace(/[\\/:*?"<>|]+/g, '_')
    .replace(/\s+/g, '_')
    .slice(0, 60) || 'table';
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 800);
}

function exportResultTable(index, format) {
  const table = APP.lastResultTables[index];
  if (!table) return toast('没有可导出的表格', 'warning');
  const { headers, rows } = tableRowsAndHeaders(table);
  if (!headers.length || !rows.length) return toast('当前表格没有可导出的数据', 'warning');
  const title = safeFilenamePart(table.title || `result_table_${index + 1}`);
  if (format === 'txt') {
    downloadBlob(new Blob([rowsToTxt(headers, rows)], { type: 'text/plain;charset=utf-8' }), `${title}.txt`);
  } else if (format === 'pdf') {
    const win = window.open('', '_blank', 'noopener,noreferrer');
    if (!win) return toast('浏览器阻止了PDF打印窗口，请允许弹出窗口后重试', 'warning');
    win.document.open();
    win.document.write(rowsToHtmlTable(headers, rows, table.title || `result_table_${index + 1}`));
    win.document.close();
    win.focus();
    setTimeout(() => win.print(), 300);
  } else {
    downloadBlob(new Blob([rowsToCsv(headers, rows)], { type: 'text/csv;charset=utf-8' }), `${title}.csv`);
  }
}

function plainDiscussionText(value) {
  return String(value || '')
    .replace(/^#{1,6}\s*/gm, '')
    .replace(/\*\*/g, '')
    .replace(/`/g, '')
    .replace(/^\s*[-*]\s+/gm, '')
    .trim();
}

function extractDiscussionSection(markdown, title) {
  const lines = String(markdown || '').split(/\r?\n/);
  const start = lines.findIndex(line => line.replace(/^#+\s*/, '').trim().startsWith(title));
  if (start < 0) return '';
  const out = [];
  for (let i = start + 1; i < lines.length; i += 1) {
    if (/^#{2,3}\s+/.test(lines[i])) break;
    if (lines[i].trim()) out.push(lines[i]);
  }
  return plainDiscussionText(out.join('\n'));
}

function compactSentence(text, fallback, limit = 120) {
  const source = plainDiscussionText(text).replace(/\s+/g, ' ');
  if (!source) return fallback;
  const sentence = source.split(/(?<=[。！？.!?])\s*/).find(Boolean) || source;
  return sentence.length > limit ? `${sentence.slice(0, limit - 1)}…` : sentence;
}

function renderDiscussionFlow(chartTitle) {
  const nodes = [
    ['数据', `${STATE.rowCount || 0} 行`],
    ['模型', getActiveMethod()?.name || '统计方法'],
    ['图表', chartTitle || '核心图'],
    ['判断', '方向与不确定性'],
  ];
  return `
    <div class="discussion-flow" aria-label="结果解读流程">
      ${nodes.map((node, index) => `
        <div class="discussion-flow-node">
          <strong>${html(node[0])}</strong>
          <span>${html(node[1])}</span>
        </div>
        ${index < nodes.length - 1 ? '<div class="discussion-flow-line"></div>' : ''}
      `).join('')}
    </div>
  `;
}

function renderDiscussion(markdown) {
  const container = dom('discussionContainer');
  if (!markdown) {
    container.innerHTML = '<div class="empty-state small">本次分析没有返回结果解释。</div>';
    return;
  }
  container.innerHTML = `
    <section class="discussion-summary">
      <div class="discussion-detail">
        <div>
          ${String(markdown).split(/\r?\n/).map(line => {
            if (!line.trim()) return '';
            if (line.startsWith('### ')) return `<h3>${html(line.slice(4))}</h3>`;
            if (line.startsWith('## ')) return `<h2>${html(line.slice(3))}</h2>`;
            if (line.startsWith('# ')) return `<h2>${html(line.slice(2))}</h2>`;
            if (/^[-*]\s+/.test(line)) return `<p class="bullet">• ${html(line.replace(/^[-*]\s+/, ''))}</p>`;
            return `<p>${html(line)}</p>`;
          }).join('')}
        </div>
      </div>
    </section>
  `;
}

function bindResultTabs() {
  domAll('.result-top-tab').forEach(button => {
    button.addEventListener('click', () => setResultTab(button.dataset.resultTab));
  });
}

function setResultTab(tab) {
  APP.activeResultTab = tab;
  domAll('.result-top-tab').forEach(button => button.classList.toggle('active', button.dataset.resultTab === tab));
  domAll('.result-view').forEach(view => view.classList.toggle('active', view.id === `result-view-${tab}`));
  if (tab === 'chart') rerenderCurrentCharts();
}

function bindChartControls() {
  dom('chartThemeSelect').addEventListener('change', event => {
    STATE.chartTheme = event.target.value;
    rerenderCurrentCharts();
  });
  dom('chartTitleInput').addEventListener('input', event => {
    STATE.chartTitle = event.target.value;
    rerenderCurrentCharts();
  });
  domAll('.export-btn').forEach(button => {
    button.addEventListener('click', () => exportCurrentChart(button.dataset.fmt));
  });
}

function renderAppearanceControls() {
  const container = dom('appearanceControls');
  if (!container) return;

  /* Detect active chart trace types */
  const activeIdx = Number.isFinite(STATE.activeChartIndex) ? STATE.activeChartIndex : 0;
  const traces = STATE.currentPlotlyData || [];
  const traceTypes = new Set();
  const traceModes = new Set();
  const traceNames = [];
  const MAX_TRACE_COLOR_CONTROLS = 20;
  traces.forEach((t, i) => {
    const type = (t && t.type) || 'scatter';
    traceTypes.add(type);
    const mode = String((t && t.mode) || '');
    if (mode.includes('markers')) traceModes.add('markers');
    if (mode.includes('lines')) traceModes.add('lines');
    if (!mode.includes('markers') && !mode.includes('lines') && type === 'scatter') {
      traceModes.add('markers');
    }
    if (traceNames.length < MAX_TRACE_COLOR_CONTROLS) {
      traceNames.push({ name: t.name || `变量${i + 1}`, index: i });
    }
  });

  const hasMarkers = traceModes.has('markers');
  const hasLines = traceModes.has('lines');
  const hasBars = traceTypes.has('bar');
  const hasBox = traceTypes.has('box');
  const hasViolin = traceTypes.has('violin');
  const hasHistogram = traceTypes.has('histogram');
  const hasPie = traceTypes.has('pie');
  const hasHeatmap = traceTypes.has('heatmap') || traceTypes.has('heatmapgl') || traceTypes.has('contour') || traceTypes.has('surface');
  const hasSankey = traceTypes.has('sankey');
  const hasScatter = traceTypes.has('scatter') || traceTypes.has('scattergl');
  const layoutShapes = Array.isArray(STATE.currentPlotlyLayout?.shapes) ? STATE.currentPlotlyLayout.shapes : [];
  const approxSame = (a, b) => Math.abs(Number(a) - Number(b)) < 1e-9;
  const isFullSpan = (a, b) => approxSame(a, 0) && approxSame(b, 1);
  const isGeneratedAxisShape = (shape) => {
    const name = String(shape?.name || '');
    return name.startsWith('v27_axis_')
      || name.startsWith('v25_axis_')
      || name.startsWith('v21_arrow_')
      || name.startsWith('v20_arrow_')
      || name === 'custom_arrow_axis_v19'
      || /(^|_)axis(_|$)/i.test(name);
  };
  const isSolidZeroBaseline = (shape) => {
    const dash = String(shape?.line?.dash || 'solid').toLowerCase();
    if (dash && dash !== 'solid') return false;
    const xref = String(shape?.xref || '');
    const yref = String(shape?.yref || '');
    const isHorizontalZero = approxSame(shape?.y0, 0)
      && approxSame(shape?.y1, 0)
      && isFullSpan(shape?.x0, shape?.x1)
      && /paper|domain/.test(xref);
    const isVerticalZero = approxSame(shape?.x0, 0)
      && approxSame(shape?.x1, 0)
      && isFullSpan(shape?.y0, shape?.y1)
      && /paper|domain/.test(yref);
    return isHorizontalZero || isVerticalZero;
  };
  const isAdjustableReferenceLine = (shape) => {
    if (!shape || shape.type !== 'line' || isGeneratedAxisShape(shape)) return false;
    return !isSolidZeroBaseline(shape);
  };
  const hasReferenceLines = layoutShapes.some(isAdjustableReferenceLine);
  const hasLabels = traces.some(t => {
    if (!t) return false;
    const type = String(t.type || '');
    return Array.isArray(t.text)
      || typeof t.text === 'string'
      || Boolean(t.texttemplate)
      || (Array.isArray(t.labels) && t.labels.length > 0)
      || (type === 'sankey' && Array.isArray(t.node?.label) && t.node.label.length > 0);
  });

  const toColorInputValue = (value, fallback = '#2563eb') => {
    const raw = String(value || '').trim();
    if (/^#[0-9a-f]{6}$/i.test(raw)) return raw;
    if (/^#[0-9a-f]{3}$/i.test(raw)) {
      return `#${raw.slice(1).split('').map(c => c + c).join('')}`;
    }
    return fallback;
  };
  const defaultCategoryColor = (index) => {
    if (typeof getChartTraceDisplayColor === 'function') {
      return toColorInputValue(getChartTraceDisplayColor(index), '#2563eb');
    }
    const fallback = ['#2E6F9E', '#D95F59', '#2A9D8F', '#E9A93A', '#6F5AA7', '#7C8B52'];
    return fallback[index % fallback.length];
  };
  const activeCategoryColors = (STATE.userBarCategoryColorsByChart || {})[activeIdx] || {};
  const categoryColorItems = [];
  const seenCategoryLabels = new Set();
  const barTraces = traces.filter(t => t && String(t.type || '') === 'bar');
  const barLabels = (t) => {
    const source = t.orientation === 'h' ? t.y : t.x;
    return (Array.isArray(source) ? source : []).map(value => String(value));
  };
  const isSingleBarPerTrace = barTraces.length > 1
    && barTraces.every(t => barLabels(t).length === 1);
  const isGroupedCategoricalBar = barTraces.length > 1
    && barTraces.every(t => barLabels(t).length > 0)
    && barTraces.every(t => barLabels(t).join('\u0001') === barLabels(barTraces[0]).join('\u0001'));
  const addCategoryColorItem = (label, color) => {
    const key = String(label ?? '').trim();
    if (!key || seenCategoryLabels.has(key)) return;
    seenCategoryLabels.add(key);
    categoryColorItems.push({
      label: key,
      color: toColorInputValue(activeCategoryColors[key] || color || defaultCategoryColor(categoryColorItems.length), defaultCategoryColor(categoryColorItems.length)),
    });
  };
  traces.forEach((t, traceIndex) => {
    if (!t) return;
    const type = String(t.type || 'scatter');
    const markerColor = t.marker && t.marker.color;
    const markerColors = Array.isArray(markerColor) ? markerColor : [];
    if (type === 'bar') {
      if (isSingleBarPerTrace) return;
      if (isGroupedCategoricalBar) return;
      const source = t.orientation === 'h' ? t.y : t.x;
      (Array.isArray(source) ? source : []).forEach((label, pointIndex) => {
        addCategoryColorItem(label, markerColors[pointIndex] || defaultCategoryColor(pointIndex));
      });
    } else if ((type === 'scatter' || type === 'scattergl') && Array.isArray(t.x)) {
      const mode = String(t.mode || 'markers');
      const hasCategoricalX = t.x.length > 1 && t.x.some(value => !Number.isFinite(Number(value)));
      const textLabels = Array.isArray(t.text) && t.text.length === t.x.length
        ? t.text.map(value => String(value ?? '').trim())
        : [];
      const pointLabels = textLabels.some(Boolean) ? textLabels : (hasCategoricalX ? t.x : []);
      if (pointLabels.length && (mode.includes('markers') || !mode.includes('lines'))) {
        pointLabels.forEach((label, pointIndex) => {
          addCategoryColorItem(label, markerColors[pointIndex] || defaultCategoryColor(pointIndex + traceIndex));
        });
      }
    }
  });

  /* Per-trace color controls */
  const activeColors = (STATE.userTraceColorsByChart || {})[activeIdx] || [];
  const activeColorsArr = Array.isArray(activeColors) ? activeColors : [];
  const traceColorHTML = traceNames.length > 0 && !hasHeatmap ? `
    <div class="color-controls-section">
      <span class="color-controls-label">变量颜色</span>
      <div class="color-buttons-grid">
        ${traceNames.map((t) => {
          const trace = traces[t.index];
          const rawColor = activeColorsArr[t.index] || (trace && trace.marker && typeof trace.marker.color === 'string' ? trace.marker.color : (trace && trace.line && trace.line.color) || '#2563eb');
          const color = toColorInputValue(rawColor, defaultCategoryColor(t.index));
          return `
          <div class="color-row" data-trace-index="${t.index}">
            <span class="color-trace-name">${escapeHtml(t.name)}</span>
            <input class="color-picker-input" data-trace-color="${t.index}" type="color" value="${escapeHtml(color)}" title="${escapeHtml(t.name)} 颜色" />
          </div>
          `;
        }).join('')}
      </div>
    </div>
  ` : '';
  const categoryColorHTML = categoryColorItems.length > 1 && !hasHeatmap ? `
    <div class="color-controls-section">
      <span class="color-controls-label">类别颜色</span>
      <div class="color-buttons-grid">
        ${categoryColorItems.map(item => `
          <div class="color-row" data-category-color-row="${escapeHtml(item.label)}">
            <span class="color-trace-name">${escapeHtml(item.label)}</span>
            <input class="color-picker-input" data-category-color="${escapeHtml(item.label)}" type="color" value="${escapeHtml(item.color)}" title="${escapeHtml(item.label)} 颜色" />
          </div>
        `).join('')}
      </div>
    </div>
  ` : '';

  /* Dynamic controls based on detected trace types */
  let dynamicControls = '';
  const hasMarkerPointControls = hasMarkers || hasBox || hasViolin || (hasScatter && !hasLines);
  const hasOpacityControls = hasMarkerPointControls || hasBars || hasHistogram;
  if (hasMarkerPointControls) {
    dynamicControls += `
      <label class="field"><span>点大小</span><input data-chart-state="markerSize" type="number" min="2" max="30" step="1" value="${Number(STATE.markerSize || 8)}" /></label>
      <label class="field"><span>点形状</span><select data-chart-state="markerShape"><option value="circle" ${(STATE.markerShape||'circle')==='circle'?'selected':''}>circle</option><option value="square" ${STATE.markerShape==='square'?'selected':''}>square</option><option value="diamond" ${STATE.markerShape==='diamond'?'selected':''}>diamond</option><option value="cross" ${STATE.markerShape==='cross'?'selected':''}>cross</option></select></label>
    `;
  }
  if (hasOpacityControls) {
    dynamicControls += `
      <label class="field"><span>透明度</span><input data-chart-state="markerOpacity" type="number" min="0.15" max="1" step="0.05" value="${Number(STATE.markerOpacity || 0.88)}" /></label>
    `;
  }
  if (hasLines) {
    dynamicControls += `
      <label class="field"><span>线宽</span><input data-chart-state="lineWidth" type="number" min="0.5" max="12" step="0.5" value="${Number(STATE.lineWidth || 2.5)}" /></label>
      <label class="field"><span>线型</span><select data-chart-state="lineDash"><option value="solid" ${(STATE.lineDash||'solid')==='solid'?'selected':''}>solid</option><option value="dash" ${STATE.lineDash==='dash'?'selected':''}>dash</option><option value="dot" ${STATE.lineDash==='dot'?'selected':''}>dot</option><option value="dashdot" ${STATE.lineDash==='dashdot'?'selected':''}>dashdot</option></select></label>
    `;
  }
  if (hasReferenceLines) {
    dynamicControls += `
      <label class="field"><span>参考线颜色</span><input data-chart-state="referenceLineColor" type="color" value="${escapeHtml(toColorInputValue(STATE.referenceLineColor || '#64748b', '#64748b'))}" /></label>
      <label class="field"><span>参考线宽</span><input data-chart-state="referenceLineWidth" type="number" min="0.5" max="10" step="0.5" value="${Number(STATE.referenceLineWidth || 1.8)}" /></label>
      <label class="field"><span>参考线线型</span><select data-chart-state="referenceLineDash">
        <option value="solid" ${(STATE.referenceLineDash||'dash')==='solid'?'selected':''}>solid</option>
        <option value="dash" ${(STATE.referenceLineDash||'dash')==='dash'?'selected':''}>dash</option>
        <option value="dot" ${STATE.referenceLineDash==='dot'?'selected':''}>dot</option>
        <option value="dashdot" ${STATE.referenceLineDash==='dashdot'?'selected':''}>dashdot</option>
      </select></label>
    `;
  }
  if (hasBars || hasHistogram) {
    dynamicControls += `
      <label class="field"><span>柱宽</span><input data-chart-state="barWidth" type="number" min="0.1" max="1" step="0.05" value="${Number(STATE.barWidth || 0.62)}" /></label>
      <label class="field"><span>柱状样式</span><select data-chart-state="barStyle">
        <option value="solid" ${(STATE.barStyle||'solid')==='solid'?'selected':''}>实心填充</option>
        <option value="outline" ${STATE.barStyle==='outline'?'selected':''}>空心描边</option>
        <option value="slash" ${STATE.barStyle==='slash'?'selected':''}>斜线条纹</option>
        <option value="backslash" ${STATE.barStyle==='backslash'?'selected':''}>反斜线条纹</option>
        <option value="cross" ${STATE.barStyle==='cross'?'selected':''}>交叉条纹</option>
        <option value="dot" ${STATE.barStyle==='dot'?'selected':''}>点状填充</option>
      </select></label>
    `;
  }
  if (hasHistogram) {
    dynamicControls += `
      <label class="field"><span>直方图分箱</span><input data-chart-state="histogramBins" type="number" min="5" max="80" step="1" value="${Number(STATE.histogramBins || 24)}" /></label>
    `;
  }
  if (hasBox) {
    dynamicControls += `
      <label class="field"><span>箱线点</span><select data-chart-state="boxPoints"><option value="outliers" ${(STATE.boxPoints||'outliers')==='outliers'?'selected':''}>outliers</option><option value="all" ${STATE.boxPoints==='all'?'selected':''}>all</option><option value="false" ${STATE.boxPoints==='false'?'selected':''}>false</option></select></label>
    `;
  }
  if (hasViolin) {
    dynamicControls += `
      <label class="field"><span>小提琴点</span><select data-chart-state="violinPoints"><option value="outliers" ${(STATE.violinPoints||'outliers')==='outliers'?'selected':''}>outliers</option><option value="all" ${STATE.violinPoints==='all'?'selected':''}>all</option><option value="false" ${STATE.violinPoints==='false'?'selected':''}>false</option></select></label>
    `;
  }
  if (hasPie) {
    dynamicControls += `
      <label class="field"><span>饼图内孔</span><input data-chart-state="pieHole" type="number" min="0" max="0.7" step="0.05" value="${Number(STATE.pieHole || 0)}" /></label>
    `;
  }
  if (hasHeatmap) {
    dynamicControls += `
      <label class="field"><span>热图透明度</span><input id="heatmapOpacityInput" data-chart-state="heatmapOpacity" type="range" min="0.15" max="1" step="0.05" value="${Number(STATE.heatmapOpacity ?? 0.88)}" /></label>
    `;
  }
  if (hasSankey) {
    dynamicControls += `
      <label class="field"><span>Sankey 间距</span><input data-chart-state="sankeyNodePad" type="number" min="4" max="40" step="1" value="${Number(STATE.sankeyNodePad || 16)}" /></label>
      <label class="field"><span>Sankey 厚度</span><input data-chart-state="sankeyNodeThickness" type="number" min="6" max="40" step="1" value="${Number(STATE.sankeyNodeThickness || 18)}" /></label>
    `;
  }

  container.innerHTML = `
    ${traceColorHTML}
    ${categoryColorHTML}
    <div class="size-grid">
      <label class="field">
        <span>图片宽度</span>
        <input id="chartWidthInput" type="number" min="360" max="2400" step="10" value="${Number(STATE.chartWidth || 760)}" />
      </label>
      <label class="field">
        <span>图片高度</span>
        <input id="chartHeightInput" type="number" min="280" max="1800" step="10" value="${Number(STATE.chartHeight || 600)}" />
      </label>
      <label class="check-field wide">
        <input id="chartSizeLink" type="checkbox" ${APP.sizeLinked ? 'checked' : ''} />
        <span>联动宽高比例</span>
      </label>
    </div>
    <label class="field">
      <span>标题字号</span>
      <input data-chart-state="chartTitleFontSize" type="number" min="10" max="40" step="1" value="${Number(STATE.chartTitleFontSize || 18)}" />
    </label>
    <div class="compact-grid">
      <label class="field">
        <span>坐标轴标题字号</span>
        <input data-chart-state="axisTitleFontSize" type="number" min="8" max="32" step="1" value="${Number(STATE.axisTitleFontSize || 13)}" />
      </label>
      <label class="field">
        <span>坐标轴刻度字号</span>
        <input data-chart-state="axisTickFontSize" type="number" min="8" max="28" step="1" value="${Number(STATE.axisTickFontSize || 11)}" />
      </label>
      ${hasLabels ? `
        <label class="field">
          <span>标签字号</span>
          <input data-chart-state="labelFontSize" type="number" min="8" max="32" step="1" value="${Number(STATE.labelFontSize || 12)}" />
        </label>
      ` : ''}
    </div>
    <div class="compact-grid">
      ${dynamicControls}
    </div>
  `;

  syncSelectValue('markerShape', STATE.markerShape || 'circle');
  syncSelectValue('lineDash', STATE.lineDash || 'solid');
  syncSelectValue('referenceLineDash', STATE.referenceLineDash || 'dash');
  syncSelectValue('barStyle', STATE.barStyle || 'solid');
  syncSelectValue('boxPoints', STATE.boxPoints || 'outliers');
  syncSelectValue('violinPoints', STATE.violinPoints || 'outliers');
  bindSizeControls();
  domAll('[data-chart-state]', container).forEach(input => {
    input.addEventListener('change', updateChartStateFromInput);
    input.addEventListener('input', updateChartStateFromInput);
  });
  /* Bind per-trace color pickers */
  domAll('[data-trace-color]', container).forEach(input => {
    input.addEventListener('input', updateTraceColorFromInput);
    input.addEventListener('change', updateTraceColorFromInput);
  });
  domAll('[data-category-color]', container).forEach(input => {
    input.addEventListener('input', updateCategoryColorFromInput);
    input.addEventListener('change', updateCategoryColorFromInput);
  });
}

function syncSelectValue(key, value) {
  const input = document.querySelector(`[data-chart-state="${key}"]`);
  if (input) input.value = value;
}

function bindSizeControls() {
  const widthInput = dom('chartWidthInput');
  const heightInput = dom('chartHeightInput');
  const link = dom('chartSizeLink');
  if (!widthInput || !heightInput || !link) return;

  const applySize = (source) => {
    let width = Number(widthInput.value || STATE.chartWidth || 760);
    let height = Number(heightInput.value || STATE.chartHeight || 600);
    if (APP.sizeLinked) {
      if (source === 'width') {
        height = Math.round(width / APP.chartAspect);
        heightInput.value = String(height);
      } else if (source === 'height') {
        width = Math.round(height * APP.chartAspect);
        widthInput.value = String(width);
      }
    }
    STATE.chartWidth = width;
    STATE.chartHeight = height;
    rerenderCurrentCharts();
  };

  link.addEventListener('change', () => {
    APP.sizeLinked = link.checked;
    APP.chartAspect = Number(widthInput.value || 760) / Math.max(1, Number(heightInput.value || 600));
  });
  widthInput.addEventListener('input', () => applySize('width'));
  heightInput.addEventListener('input', () => applySize('height'));
}

function updateChartStateFromInput(event) {
  const input = event.currentTarget;
  const key = input.dataset.chartState;
  if (!key) return;
  STATE[key] = (input.type === 'number' || input.type === 'range') ? Number(input.value) : input.value;
  rerenderCurrentCharts();
}

function updateTraceColorFromInput(event) {
  const input = event.currentTarget;
  const traceIndex = Number(input.dataset.traceColor);
  if (!Number.isFinite(traceIndex)) return;
  const activeIdx = Number.isFinite(STATE.activeChartIndex) ? STATE.activeChartIndex : 0;
  if (!STATE.userTraceColorsByChart) STATE.userTraceColorsByChart = {};
  if (!STATE.userTraceColorsByChart[activeIdx]) STATE.userTraceColorsByChart[activeIdx] = [];
  STATE.userTraceColorsByChart[activeIdx][traceIndex] = input.value;
  rerenderCurrentCharts();
}

function updateCategoryColorFromInput(event) {
  const input = event.currentTarget;
  const label = input.dataset.categoryColor;
  if (!label) return;
  const activeIdx = Number.isFinite(STATE.activeChartIndex) ? STATE.activeChartIndex : 0;
  if (!STATE.userBarCategoryColorsByChart) STATE.userBarCategoryColorsByChart = {};
  if (!STATE.userBarCategoryColorsByChart[activeIdx]) STATE.userBarCategoryColorsByChart[activeIdx] = {};
  STATE.userBarCategoryColorsByChart[activeIdx][label] = input.value;
  rerenderCurrentCharts();
}

function rerenderCurrentCharts() {
  if (APP.lastResult && typeof renderAllCharts === 'function') {
    renderAllCharts(APP.lastResult.charts || []);
  }
}

function cloneSerializable(value, fallback) {
  try {
    return JSON.parse(JSON.stringify(value ?? fallback));
  } catch (error) {
    return fallback;
  }
}

function parsePlotlyFigure(value) {
  if (!value) return {};
  if (typeof value === 'string') {
    try {
      return JSON.parse(value);
    } catch (error) {
      return {};
    }
  }
  return cloneSerializable(value, {});
}

function activeChartDefinition() {
  const charts = STATE.currentChartBundle || APP.lastResult?.charts || [];
  if (!charts.length) return null;
  const index = Math.max(0, Math.min(Number(STATE.activeChartIndex || 0), charts.length - 1));
  return charts[index];
}

function buildChartExportFigure(plot) {
  const active = activeChartDefinition();
  const original = parsePlotlyFigure(active?.plotly);
  const data = cloneSerializable(STATE.currentPlotlyData || plot?.data || original.data || [], []);
  const layout = cloneSerializable(STATE.currentPlotlyLayout || plot?.layout || original.layout || {}, {});
  layout.width = Number(STATE.chartWidth || layout.width || 760);
  layout.height = Number(STATE.chartHeight || layout.height || 600);
  layout.autosize = false;
  if (STATE.chartTitle) {
    layout.title = typeof layout.title === 'object' ? { ...layout.title, text: STATE.chartTitle } : { text: STATE.chartTitle };
  }
  return { data, layout };
}

async function exportChartViaBackend(format, plot) {
  const figure = buildChartExportFigure(plot);
  if (!figure.data.length) return toast('当前图表没有可导出的数据', 'warning');
  const filename = `mlhigh_${safeFilenamePart(STATE.activeMethodId || 'chart')}_${Date.now()}`;
  const response = await fetch('/api/export/chart-image', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      format,
      filename,
      width: Number(STATE.chartWidth || figure.layout.width || 760),
      height: Number(STATE.chartHeight || figure.layout.height || 600),
      figure,
    }),
  });
  if (!response.ok) {
    let message = response.statusText || '导出失败';
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch (error) {
      message = await response.text() || message;
    }
    throw new Error(message);
  }
  const blob = await response.blob();
  downloadBlob(blob, `${filename}.${format}`);
}

async function exportCurrentChart(format) {
  const plot = document.querySelector('#chartPreviewContainer .js-plotly-plot');
  if (format === 'csv') {
    exportChartDataCsv();
    return;
  }
  if (!plot || !window.Plotly) {
    toast('当前没有可导出的图表', 'warning');
    return;
  }
  const filename = `mlhigh_${STATE.activeMethodId || 'chart'}_${Date.now()}`;
  try {
    if (format === 'pdf') {
      await exportChartViaBackend('pdf', plot);
      return;
    }
    await Plotly.downloadImage(plot, {
      format,
      filename,
      width: Number(STATE.chartWidth || 760),
      height: Number(STATE.chartHeight || 600),
      scale: format === 'png' ? 2 : 1,
    });
  } catch (error) {
    toast(`导出失败：${error.message}`, 'error');
  }
}

function exportChartDataCsv() {
  const traces = STATE.currentPlotlyData || [];
  if (!traces.length) return toast('当前图表没有可导出的数据', 'warning');
  const rows = [['trace', 'x', 'y']];
  traces.forEach((trace, traceIndex) => {
    const xs = Array.isArray(trace.x) ? trace.x : [];
    const ys = Array.isArray(trace.y) ? trace.y : [];
    const n = Math.max(xs.length, ys.length);
    for (let i = 0; i < n; i += 1) rows.push([trace.name || `trace_${traceIndex + 1}`, xs[i] ?? '', ys[i] ?? '']);
  });
  const csv = rows.map(row => row.map(cell => `"${String(cell).replaceAll('"', '""')}"`).join(',')).join('\n');
  downloadBlob(new Blob([csv], { type: 'text/csv;charset=utf-8' }), `mlhigh_${safeFilenamePart(STATE.activeMethodId || 'chart')}_data.csv`);
}

window.renderAppearanceControls = renderAppearanceControls;

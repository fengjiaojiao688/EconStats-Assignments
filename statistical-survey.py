# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import joblib

# 设置图表风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用于显示负号

# 加载数据
df = pd.read_excel("宠物殡葬处理后数据(1).xlsx")

# 数据预处理优化 =============================================================
categorical_cols = ['age', 'gender', 'pet_owner', 'income']
continuous_cols = ['emotion', 'need_pro', 'eco_friendly_preference', 'individual_cremation_importance',
                   'ashes_memorabilia_importance', 'farewell_ceremony_importance', 'online_memorial_importance',
                   'additional_services_demand', 'privacy_protection_importance', 'brand_reputation_importance',
                  ]

# 使用更兼容的方法处理分类变量
# 创建副本避免修改原始数据
processed_df = df.copy()

# 对分类变量进行one-hot编码
for col in categorical_cols:
    dummies = pd.get_dummies(processed_df[col], prefix=col, drop_first=True)
    processed_df = pd.concat([processed_df, dummies], axis=1)
    processed_df.drop(col, axis=1, inplace=True)

# 对连续变量进行标准化
scaler = StandardScaler()
processed_df[continuous_cols] = scaler.fit_transform(processed_df[continuous_cols])

# 填充缺失值
processed_df.fillna(processed_df.median(), inplace=True)

# 划分自变量和因变量
X = processed_df.drop(['序号', 'payment_willingness'], axis=1, errors='ignore')
y = processed_df['payment_willingness']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 模型构建与优化 =============================================================
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)

# 简化的网格搜索（根据数据量调整）
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=3,  # 减少交叉验证折数以加快计算
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳模型R²: {grid_search.best_score_:.4f}")

# 模型评估 =================================================================
y_pred = best_rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n模型评估结果:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"平均绝对误差 (MAE): {mae:.4f}")
print(f"决定系数 (R²): {r2:.4f}")

# 预测值与实际值对比图
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('实际支付意愿')
plt.ylabel('预测支付意愿')
plt.title('预测值 vs 实际值')
plt.savefig('预测值与实际值对比.png', dpi=300, bbox_inches='tight')
plt.show()

# 特征重要性分析 ===========================================================

# 1. 内置特征重要性
importances = best_rf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'特征': feature_names, '重要性': importances})
feature_importance_df = feature_importance_df.sort_values('重要性', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='重要性', y='特征', data=feature_importance_df.head(15))
plt.title('Top 15 特征重要性')
plt.tight_layout()
plt.savefig('特征重要性.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 排列重要性 - 更可靠的特征重要性评估
print("\n计算排列重要性...")
result = permutation_importance(
    best_rf, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
)

# 修复错误：正确处理排列重要性的结果
perm_importances = result.importances_mean
perm_std = result.importances_std

# 创建DataFrame并排序
perm_importance_df = pd.DataFrame({
    '特征': feature_names,
    '重要性': perm_importances,
    '标准差': perm_std
}).sort_values('重要性', ascending=False)

# 只取前15个特征
perm_importance_top = perm_importance_df.head(15)

# 修复错误：正确绘制带误差线的条形图
plt.figure(figsize=(12, 8))
plt.barh(
    perm_importance_top['特征'],
    perm_importance_top['重要性'],
    xerr=perm_importance_top['标准差'],
    capsize=5
)
plt.xlabel('排列重要性')
plt.ylabel('特征')
plt.title('Top 15 排列特征重要性')
plt.gca().invert_yaxis()  # 反转y轴使最重要的特征在顶部
plt.tight_layout()
plt.savefig('排列特征重要性.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 部分依赖图 (PDP) - 分析特征与目标的关系
print("\n生成部分依赖图...")
top_features = perm_importance_df.head(3)['特征'].tolist()

for feature in top_features:
    try:
        # 使用sklearn的PartialDependenceDisplay
        fig, ax = plt.subplots(figsize=(10, 6))
        PartialDependenceDisplay.from_estimator(
            best_rf, X_train, [feature], ax=ax
        )
        plt.title(f"'{feature}'的部分依赖图")
        plt.ylabel("支付意愿")
        plt.tight_layout()
        plt.savefig(f'{feature}_部分依赖图.png', dpi=300, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"无法为特征 '{feature}' 生成部分依赖图: {str(e)}")

# 4. 特征分布分析
print("\n分析特征分布...")
plt.figure(figsize=(12, 8))
for i, feature in enumerate(top_features[:3]):
    plt.subplot(3, 1, i + 1)
    # 检查特征是否在原始数据中
    if feature in df.columns:
        sns.histplot(df[feature], kde=True)
    else:
        # 可能是one-hot编码的特征
        sns.histplot(X[feature], kde=True)
    plt.title(f"'{feature}'的分布")
plt.tight_layout()
plt.savefig('特征分布.png', dpi=300, bbox_inches='tight')
plt.show()

# 模型保存与部署 ============================================================
joblib.dump(best_rf, '宠物殡葬支付意愿预测模型.pkl')
joblib.dump(scaler, '标准化器.pkl')

# 保存特征名称
with open('模型特征名称.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(X.columns.tolist()))

print("\n模型和预处理管道已保存！")

# 业务洞察与建议 ============================================================
print("\n=== 业务洞察与建议 ===")
print("1. 最重要的影响因素:")
for i, row in perm_importance_df.head(5).iterrows():
    print(f"  - {row['特征']}: 重要性={row['重要性']:.4f} ± {row['标准差']:.4f}")

print("\n2. 基于分析的关键发现:")
if not perm_importance_df.empty:
    top_feature = perm_importance_df.iloc[0]['特征']
    print(f"  最重要的特征是 '{top_feature}'，它解释了支付意愿的大部分变化")

    # 根据特征名称提供业务解释
    if 'eco' in top_feature.lower():
        print("  这表明环保因素对消费者决策有重大影响")
    elif 'price' in top_feature.lower() or 'cost' in top_feature.lower():
        print("  这表明价格敏感度是影响支付意愿的关键因素")
    elif 'brand' in top_feature.lower():
        print("  这表明品牌声誉在消费者决策中起关键作用")
    elif 'emotion' in top_feature.lower():
        print("  这表明情感因素对支付意愿有强烈影响")
    else:
        print("  请进一步分析此特征与支付意愿的关系")

print("\n3. 营销策略建议:")
print("  - 针对高支付意愿客户群体:")
print(f"      * 识别特征: {', '.join(perm_importance_df.head(3)['特征'].tolist())}")
print("  - 优化服务套餐:")
print("      * 强化对支付意愿影响最大的服务要素")
print("  - 价格策略:")
print("      * 根据价格敏感度特征设计分层定价")
print("  - 营销重点:")
print("      * 突出环保、情感价值和品牌可信度等关键因素")

# 保存业务洞察报告
with open('业务洞察报告.txt', 'w', encoding='utf-8') as report:
    report.write("=== 宠物殡葬支付意愿分析报告 ===\n\n")
    report.write("1. 模型性能:\n")
    report.write(f"   - R²: {r2:.4f}\n")
    report.write(f"   - MSE: {mse:.4f}\n")
    report.write(f"   - MAE: {mae:.4f}\n\n")

    report.write("2. 关键影响因素:\n")
    for i, row in perm_importance_df.head(5).iterrows():
        report.write(f"   - {row['特征']}: 重要性={row['重要性']:.4f} ± {row['标准差']:.4f}\n")

    report.write("\n3. 业务建议:\n")
    report.write("   - 重点优化对支付意愿影响最大的服务要素\n")
    report.write("   - 针对高价值客户群体设计专属套餐\n")
    report.write("   - 加强品牌建设和环保服务宣传\n")

print("\n分析报告已保存为 '业务洞察报告.txt'")

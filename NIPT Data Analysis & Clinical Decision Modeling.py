# === 可直接运行的求解代码（Python 3.x）===
# 依赖库（如本环境未安装，请在本地按需安装）：
# pip install numpy pandas matplotlib scikit-learn statsmodels openpyxl
#
# 说明：
# 1) 所有图均使用 matplotlib，且每个图单独绘制、不指定颜色，满足题目要求。
# 2) 代码包含“数据输入 → 参数初始化 → 模型调用 → 结果输出”的结构化流程与关键注释。
# 3) 若数据表头为中文，本代码将自动识别并通过“字段别名”字典进行适配；若列名存在差异，可在 aliases 中手动补齐。

import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.cluster import KMeans

import statsmodels.api as sm

#=============================#
# 一、数据输入
#=============================#

DATA_PATH = "/mnt/data/附件.xlsx"  # 题目附件
assert os.path.exists(DATA_PATH), "未找到附件.xlsx，请确认路径为 /mnt/data/附件.xlsx"

# 尝试读取 Excel（默认读取第一张表）
df = pd.read_excel(DATA_PATH, sheet_name=0)

# 为了便于后续处理，创建“字段别名”适配（不同附件可能中文有轻微差异）
aliases = {
    "样本序号": "sample_id",
    "孕妇代码": "patient_id",
    "孕妇年龄": "age",
    "孕妇身高": "height",
    "孕妇体重": "weight",
    "末次月经时间": "last_period_date",
    "IVF 妊娠方式": "ivf",
    "检测时间": "test_time",
    "检测抽血次数": "draw_count",
    "孕妇本次检测时的孕周（周数+天数）": "gest_weeks_text",
    "孕妇 BMI 指标": "bmi",
    "原始测序数据的总读段数（个）": "reads_total",
    "总读段数中在参考基因组上比对的比例": "map_ratio",
    "总读段数中重复读段的比例": "dup_ratio",
    "总读段数中唯一比对的读段数（个）": "uniq_reads",
    "GC 含量": "gc_total",
    "13 号染色体的 Z 值": "z13",
    "18 号染色体的 Z 值": "z18",
    "21 号染色体的 Z 值": "z21",
    "X 染色体的 Z 值": "zx",
    "Y 染色体的 Z 值（女胎数据此列为空白）": "zy",
    "Y 染色体浓度，即 Y 染色体游离 DNA 片段的比例（女胎数据此列为空白）": "y_frac",
    "X 染色体浓度（其数值是通过生物信息学在一定假设下通过数据分析估计得出，可能出现负值）": "x_frac",
    "13 号染色体的 GC 含量": "gc13",
    "18 号染色体的 GC 含量": "gc18",
    "21 号染色体的 GC 含量": "gc21",
    "被过滤掉的读段数占总读段数的比例": "filter_ratio",
    "检测出的 13 号，18 号，21 号染色体非整倍体，即数量异常，空白即为无异常": "aneu_tag",
    "孕妇的怀孕次数": "gravidity",
    "孕妇的生产次数": "parity",
    "胎儿是否健康（婴儿出生后的结果）": "outcome"
}

# 将 DataFrame 列名映射为英文友好名（若不存在对应别名则保留原名）
rename_map = {}
for col in df.columns:
    rename_map[col] = aliases.get(col, col)
df = df.rename(columns=rename_map)

# 显示前几行，帮助确认字段是否正确匹配
import caas_jupyter_tools as cj
cj.display_dataframe_to_user("原始数据预览（前50行）", df.head(50))

#=============================#
# 二、通用预处理工具
#=============================#

def parse_gest_weeks(x):
    """
    将“孕周（周数+天数）”解析为十进制周（float）。
    可接受：'12+3', '12周+3天', 12.5, 12, NaN 等。
    规则：周 + 天/7；若为数值直接返回；若无法解析则返回 NaN。
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    # 先匹配'12+3'或'12 + 3'样式
    m = re.match(r"^\s*(\d+)\s*\+\s*(\d+)\s*$", s)
    if m:
        w, d = int(m.group(1)), int(m.group(2))
        return w + d/7.0
    # 匹配'12周+3天'样式
    m = re.match(r"^\s*(\d+)\s*周\s*\+\s*(\d+)\s*天\s*$", s)
    if m:
        w, d = int(m.group(1)), int(m.group(2))
        return w + d/7.0
    # 直接匹配整数或小数
    m = re.match(r"^\s*(\d+(\.\d+)?)\s*$", s)
    if m:
        return float(m.group(1))
    return np.nan

# 派生字段：gest_weeks (十进制周)、is_male、达标 y>=4%
df["gest_weeks"] = df.get("gest_weeks_text", np.nan).apply(parse_gest_weeks) if "gest_weeks_text" in df.columns else np.nan
if "y_frac" not in df.columns and "Y 染色体浓度" in df.columns:
    df["y_frac"] = df["Y 染色体浓度"]

df["is_male"] = df.get("y_frac").notna()
df["y_hit"] = (df.get("y_frac") >= 0.04).astype("float") if "y_frac" in df.columns else np.nan

# 筛选可用样本（有孕周、BMI）
df_use = df.copy()
if "bmi" not in df_use.columns and "孕妇 BMI 指标" in df.columns:
    df_use["bmi"] = df["孕妇 BMI 指标"]

#=============================#
# 三、问题1：Y浓度 ~ 孕周 + BMI 回归/分层模型（基础版）
#=============================#

q1_dir = "/mnt/data/q1"
os.makedirs(q1_dir, exist_ok=True)

q1 = df_use[(df_use["is_male"] == True) & df_use["gest_weeks"].notna() & df_use["bmi"].notna()]
q1 = q1[(q1["gest_weeks"] >= 10) & (q1["gest_weeks"] <= 25)]

# 注意：回归使用 OLS；若存在非线性，可加入二次项或交互项
def fit_q1_ols(data):
    X = pd.DataFrame({
        "const": 1.0,
        "gest_weeks": data["gest_weeks"].astype(float),
        "bmi": data["bmi"].astype(float),
        "gw_bmi": data["gest_weeks"].astype(float) * data["bmi"].astype(float)  # 交互项，提升拟合
    })
    y = data["y_frac"].astype(float)
    model = sm.OLS(y, X, missing="drop").fit()
    return model

q1_model = None
if len(q1) >= 30 and q1["y_frac"].notna().sum() >= 30:
    q1_model = fit_q1_ols(q1.dropna(subset=["y_frac"]))

# —— 可视化：不同 BMI 分位下，Y 浓度随孕周变化 —— #
if q1_model is not None:
    # 使用分位数作为“代表 BMI”，避免随意指定；np.mean()确保浮点平均，避免整数除法问题。
    bmi_qs = q1["bmi"].quantile([0.25, 0.5, 0.75]).values
    gw_grid = np.linspace(10, 25, 100)
    plt.figure()
    for b in bmi_qs:
        Xp = pd.DataFrame({
            "const": np.ones_like(gw_grid),
            "gest_weeks": gw_grid,
            "bmi": np.full_like(gw_grid, b),
            "gw_bmi": gw_grid * b
        })
        yhat = q1_model.predict(Xp)
        plt.plot(gw_grid, yhat, label=f"BMI≈{b:.1f}")
    plt.axhline(0.04, linestyle="--", label="阈值 4%")
    plt.xlabel("孕周（周）")
    plt.ylabel("Y 染色体浓度")
    plt.title("问题1：不同BMI下 Y 浓度-孕周 拟合曲线（OLS+交互）")
    plt.legend()
    plt.tight_layout()
    q1_plot1 = os.path.join(q1_dir, "q1_fit_curves.png")
    plt.savefig(q1_plot1, dpi=150)
    plt.close()

    # 输出模型摘要到文本
    with open(os.path.join(q1_dir, "q1_ols_summary.txt"), "w", encoding="utf-8") as f:
        f.write(str(q1_model.summary()))

#=============================#
# 四、问题2：BMI分组 + 最佳时点（固定分组 & 聚类对比）
#=============================#

q2_dir = "/mnt/data/q2"
os.makedirs(q2_dir, exist_ok=True)

# —— 方案A：固定分组（按经验或分位数微调） —— #
# 若题目强调高 BMI 人群，可设置高端细分；此处按近似经验分组，可根据数据分布自动裁剪
bins_fixed = [0, 24, 28, 32, 36, 100]  # 注意：可根据数据分布微调；必须覆盖样本范围
labels_fixed = [f"{bins_fixed[i]}~{bins_fixed[i+1]}" for i in range(len(bins_fixed)-1)]

q2 = q1.copy()
q2["bmi_group_fixed"] = pd.cut(q2["bmi"], bins=bins_fixed, labels=labels_fixed, include_lowest=True, right=False)

# 若 Q1 模型不可用，则退化为组内线性回归（Y~GW）
def group_cross_week_estimate(data, bmi_val, model=None, safety_z=1.28):
    """
    估计在给定 BMI 下 Y 达到 4% 的最早孕周。
    若提供 OLS 模型，则用模型求解；否则用数据组内回归。
    安全裕度：t* = t_cross + z * sigma / |slope_gw|，降低误判风险。
    """
    threshold = 0.04
    if model is not None:
        # 求：beta0 + beta1*gw + beta2*bmi + beta3*gw*bmi = 0.04
        params = model.params
        b0, b1, b2, b3 = params["const"], params["gest_weeks"], params["bmi"], params["gw_bmi"]
        # 等价于：(b1 + b3*bmi)*gw + (b0 + b2*bmi - threshold) = 0
        slope = (b1 + b3*bmi_val)
        intercept = (b0 + b2*bmi_val - threshold)
        if abs(slope) < 1e-6:
            return np.nan
        t_cross = - intercept / slope
        # 残差标准差（RMSE）
        resid = model.resid
        sigma = np.sqrt(np.mean(resid**2))
        # 注意：若斜率为负（理论上应为正），此处加安全裕度会反向，需要取绝对值保证健壮
        t_opt = t_cross + safety_z * sigma / max(abs(slope), 1e-6)
        return t_cross, t_opt, slope, sigma
    else:
        # 组内回归：Y ~ a + b * GW
        sub = data.dropna(subset=["y_frac", "gest_weeks"]).copy()
        if len(sub) < 10:
            return np.nan
        X = sm.add_constant(sub["gest_weeks"].astype(float))
        y = sub["y_frac"].astype(float)
        m = sm.OLS(y, X).fit()
        b0, b1 = m.params["const"], m.params["gest_weeks"]
        slope = b1
        intercept = b0 - threshold
        if abs(slope) < 1e-6:
            return np.nan
        t_cross = -intercept / slope
        resid = m.resid
        sigma = np.sqrt(np.mean(resid**2))
        t_opt = t_cross + 1.28 * sigma / max(abs(slope), 1e-6)
        return t_cross, t_opt, slope, sigma

# 计算每个固定分组的最优时点（裁剪到[10,25]）
results_q2_fixed = []
for g, sub in q2.groupby("bmi_group_fixed"):
    if g is pd.NA or g is None:
        continue
    bmi_mean = sub["bmi"].mean()
    est = group_cross_week_estimate(q2, bmi_mean, model=q1_model)
    if isinstance(est, tuple):
        t_cross, t_opt, slope, sigma = est
    else:
        t_cross, t_opt, slope, sigma = (np.nan, np.nan, np.nan, np.nan)
    # 将时点限制在可检测区间内
    def clamp_week(w):
        if pd.isna(w):
            return np.nan
        return min(25.0, max(10.0, w))
    results_q2_fixed.append({
        "bmi_group": str(g),
        "bmi_mean": bmi_mean,
        "t_cross_est": clamp_week(t_cross),
        "t_opt_safe": clamp_week(t_opt),
        "slope_gw": slope,
        "rmse": sigma
    })

df_q2_fixed = pd.DataFrame(results_q2_fixed).sort_values("bmi_mean")

# —— 方案B：BMI 聚类分组（KMeans） —— #
# 注意：KMeans 的 n_clusters 需在 [2,8] 范围内试验；太多会导致样本过少。
n_clusters = 4
kmeans_model = None
df_q2_k = None
if len(q2) >= n_clusters:
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    q2["bmi_group_k"] = kmeans_model.fit_predict(q2[["bmi"]])
    # 计算每个聚类组的 t_opt
    rows = []
    for k, sub in q2.groupby("bmi_group_k"):
        bmi_mean = sub["bmi"].mean()
        est = group_cross_week_estimate(q2, bmi_mean, model=q1_model)
        if isinstance(est, tuple):
            t_cross, t_opt, slope, sigma = est
        else:
            t_cross, t_opt, slope, sigma = (np.nan, np.nan, np.nan, np.nan)
        rows.append({
            "cluster": int(k),
            "bmi_mean": bmi_mean,
            "t_cross_est": min(25.0, max(10.0, t_cross)) if not pd.isna(t_cross) else np.nan,
            "t_opt_safe": min(25.0, max(10.0, t_opt)) if not pd.isna(t_opt) else np.nan,
            "slope_gw": slope,
            "rmse": sigma
        })
    df_q2_k = pd.DataFrame(rows).sort_values("bmi_mean")

# —— 可视化：固定分组的最优时点 —— #
if df_q2_fixed is not None and len(df_q2_fixed) > 0:
    plt.figure()
    plt.bar(df_q2_fixed["bmi_group"].astype(str), df_q2_fixed["t_opt_safe"])
    plt.xlabel("BMI 分组（固定）")
    plt.ylabel("最佳 NIPT 时点（周）")
    plt.title("问题2：固定分组的最佳 NIPT 时点（安全裕度）")
    plt.tight_layout()
    q2_plot1 = os.path.join(q2_dir, "q2_fixed_opt_week.png")
    plt.savefig(q2_plot1, dpi=150)
    plt.close()

# 输出结果表（固定+聚类）
if df_q2_fixed is not None:
    df_q2_fixed.to_csv(os.path.join(q2_dir, "q2_fixed_groups.csv"), index=False, encoding="utf-8-sig")
if df_q2_k is not None:
    df_q2_k.to_csv(os.path.join(q2_dir, "q2_kmeans_groups.csv"), index=False, encoding="utf-8-sig")

#=============================#
# 五、问题3：多因素 Logistic 概率模型 + 达标周的阈值反推
#=============================#

q3_dir = "/mnt/data/q3"
os.makedirs(q3_dir, exist_ok=True)

q3 = q1.copy()  # 仅男胎
# 选择特征：孕周、BMI、年龄、身高、体重（根据数据可用性自动筛）
feat_candidates = ["gest_weeks", "bmi", "age", "height", "weight"]
feats = [c for c in feat_candidates if c in q3.columns]
q3 = q3.dropna(subset=["y_hit"] + feats)

logit_model = None
scaler = None
prob_target = 0.9  # 目标达标比例阈值，可在[0.7, 0.95]之间调参

if q3["y_hit"].nunique() >= 2 and len(q3) >= 50:
    X = q3[feats].astype(float).values
    y = q3["y_hit"].astype(int).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # class_weight="balanced" 让模型在类别不均衡时更稳健
    logit_model = LogisticRegression(max_iter=200, class_weight="balanced", solver="lbfgs")
    logit_model.fit(Xs, y)

    # —— 反推：按 BMI 分组，求最早周数使 P(y>=4%) ≥ prob_target —— #
    if df_q2_fixed is not None and len(df_q2_fixed) > 0:
        weeks = np.linspace(10, 25, 151)
        group_min_week = []
        for _, row in df_q2_fixed.iterrows():
            bmi_mean = row["bmi_mean"]
            # 其余变量用全体中位数填充，避免引入偏差
            age_med = np.nanmedian(q3["age"]) if "age" in q3.columns else 30.0
            h_med = np.nanmedian(q3["height"]) if "height" in q3.columns else 160.0
            w_med = np.nanmedian(q3["weight"]) if "weight" in q3.columns else 65.0
            min_w = np.nan
            for w in weeks:
                vec = []
                for f in feats:
                    if f == "gest_weeks":
                        vec.append(w)
                    elif f == "bmi":
                        vec.append(bmi_mean)
                    elif f == "age":
                        vec.append(age_med)
                    elif f == "height":
                        vec.append(h_med)
                    elif f == "weight":
                        vec.append(w_med)
                xs = scaler.transform(np.array(vec).reshape(1, -1))
                p = logit_model.predict_proba(xs)[0,1]
                if p >= prob_target:
                    min_w = w
                    break
            group_min_week.append((row["bmi_group"], bmi_mean, min_w))

        df_q3_minweek = pd.DataFrame(group_min_week, columns=["bmi_group", "bmi_mean", "week_for_prob_target"]).sort_values("bmi_mean")
        df_q3_minweek.to_csv(os.path.join(q3_dir, "q3_week_for_p_target.csv"), index=False, encoding="utf-8-sig")

        # 可视化：各组达 90% 概率的最早孕周
        plt.figure()
        plt.bar(df_q3_minweek["bmi_group"].astype(str), df_q3_minweek["week_for_prob_target"])
        plt.xlabel("BMI 分组（固定）")
        plt.ylabel(f"达到 {prob_target*100:.0f}% 达标概率的最早孕周")
        plt.title("问题3：多因素Logistic模型反推的最佳检测时点")
        plt.tight_layout()
        q3_plot1 = os.path.join(q3_dir, "q3_prob_target_week.png")
        plt.savefig(q3_plot1, dpi=150)
        plt.close()

#=============================#
# 六、问题4：女胎异常判定（基线+集成）
#=============================#

q4_dir = "/mnt/data/q4"
os.makedirs(q4_dir, exist_ok=True)

# 女胎：y_frac 为空或 is_male==False
q4 = df_use[(df_use["is_male"] == False) | (df_use.get("y_frac").isna())].copy()

# 标签：AB 列非空为异常（1），否则 0
if "aneu_tag" in q4.columns:
    y_clf = q4["aneu_tag"].notna().astype(int)
else:
    # 若缺少标签，则尝试用 z21/z18/z13 的绝对值>3 作为弱标签（仅用于演示）
    z_cols = [c for c in ["z13","z18","z21"] if c in q4.columns]
    if len(z_cols) > 0:
        y_clf = (q4[z_cols].abs().max(axis=1) > 3).astype(int)
    else:
        y_clf = pd.Series(np.zeros(len(q4), dtype=int), index=q4.index)

# 特征集合：Z 值、GC 含量、x_frac、BMI、测序读段相关
feature_cols = [c for c in ["z13","z18","z21","zx","x_frac","gc13","gc18","gc21","gc_total","bmi","reads_total","map_ratio","dup_ratio","uniq_reads","filter_ratio"] if c in q4.columns]
q4_feat = q4[feature_cols].copy()
q4_feat = q4_feat.fillna(q4_feat.median(numeric_only=True))  # 用中位数填补，稳健

# 仅在有正负样本时训练
clf_report_txt = os.path.join(q4_dir, "q4_report.txt")
if y_clf.nunique() == 2 and feature_cols:
    X_train, X_test, y_train, y_test = train_test_split(q4_feat.values, y_clf.values, test_size=0.3, random_state=42, stratify=y_clf.values)

    # 基线：LDA（适合特征近似线性可分）
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_pred_lda = lda.predict(X_test)
    y_proba_lda = lda.predict_proba(X_test)[:,1]
    auc_lda = roc_auc_score(y_test, y_proba_lda)

    # 集成：GradientBoostingClassifier（无需外部依赖，能拟合非线性）
    gbc = GradientBoostingClassifier(random_state=42)
    gbc.fit(X_train, y_train)
    y_pred_gbc = gbc.predict(X_test)
    y_proba_gbc = gbc.predict_proba(X_test)[:,1]
    auc_gbc = roc_auc_score(y_test, y_proba_gbc)

    # 输出评估报告
    with open(clf_report_txt, "w", encoding="utf-8") as f:
        f.write("=== 基线：LDA ===\n")
        f.write(classification_report(y_test, y_pred_lda))
        f.write(f"AUC: {auc_lda:.3f}\n\n")
        f.write("=== 集成：GradientBoosting ===\n")
        f.write(classification_report(y_test, y_pred_gbc))
        f.write(f"AUC: {auc_gbc:.3f}\n")

    # 绘制 ROC 曲线（单图）
    fpr1, tpr1, _ = roc_curve(y_test, y_proba_lda)
    fpr2, tpr2, _ = roc_curve(y_test, y_proba_gbc)
    plt.figure()
    plt.plot(fpr1, tpr1, label=f"LDA AUC={auc_lda:.3f}")
    plt.plot(fpr2, tpr2, label=f"GBC AUC={auc_gbc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--", label="随机")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("问题4：女胎异常判定 ROC 对比")
    plt.legend()
    plt.tight_layout()
    q4_plot1 = os.path.join(q4_dir, "q4_roc.png")
    plt.savefig(q4_plot1, dpi=150)
    plt.close()

    # 混淆矩阵（以 GBC 为例）
    cm = confusion_matrix(y_test, y_pred_gbc)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("问题4：GradientBoosting 混淆矩阵")
    plt.xlabel("预测")
    plt.ylabel("真实")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, int(v), ha="center", va="center")
    plt.tight_layout()
    q4_plot2 = os.path.join(q4_dir, "q4_confusion.png")
    plt.savefig(q4_plot2, dpi=150)
    plt.close()
else:
    with open(clf_report_txt, "w", encoding="utf-8") as f:
        f.write("标签类别不足或特征缺失，无法训练分类器；请检查 AB 列或 Z 值列。")

#=============================#
# 七、关键结果的可视化与导出
#=============================#

outputs = {}

# 问题1
if q1_model is not None:
    outputs["Q1_拟合曲线"] = q1_plot1
    outputs["Q1_模型摘要"] = os.path.join(q1_dir, "q1_ols_summary.txt")

# 问题2
if df_q2_fixed is not None and len(df_q2_fixed) > 0:
    outputs["Q2_固定分组表"] = os.path.join(q2_dir, "q2_fixed_groups.csv")
    outputs["Q2_固定分组最优时点图"] = q2_plot1
if df_q2_k is not None and len(df_q2_k) > 0:
    outputs["Q2_聚类分组表"] = os.path.join(q2_dir, "q2_kmeans_groups.csv")

# 问题3
if os.path.exists(os.path.join(q3_dir, "q3_week_for_p_target.csv")):
    outputs["Q3_达标阈值反推表"] = os.path.join(q3_dir, "q3_week_for_p_target.csv")
    outputs["Q3_最早孕周柱状图"] = os.path.join(q3_dir, "q3_prob_target_week.png")

# 问题4
outputs["Q4_评估报告"] = clf_report_txt
if os.path.exists(q4_plot1):
    outputs["Q4_ROC曲线"] = q4_plot1
if os.path.exists(q4_plot2):
    outputs["Q4_混淆矩阵"] = q4_plot2

# 打印输出清单
print("=== 关键输出文件清单 ===")
for k, v in outputs.items():
    print(f"{k}: {v}")

# 在界面展示关键表格（若存在）
if df_q2_fixed is not None and len(df_q2_fixed) > 0:
    cj.display_dataframe_to_user("问题2-固定分组与最优时点（预览）", df_q2_fixed)
if 'df_q3_minweek' in globals():
    cj.display_dataframe_to_user("问题3-达到目标概率的最早孕周（预览）", df_q3_minweek)
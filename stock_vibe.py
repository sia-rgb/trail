import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === 1. 生成模拟数据 (Mock Data) ===
print("🧪 正在启动实验室模式，生成模拟数据...")

# 设定参数
np.random.seed(42)  # 固定随机种子，保证每次运行结果一致 (也可以注释掉这行，每次看不同的结果)
days = 252          # 模拟 1 年
start_price = 200   # 起始价
mu = 0.0005         # 假定的每日平均收益率 (漂移项)
sigma = 0.02        # 假定的每日波动 (约 2%)

# 生成日期索引
dates = pd.date_range(start='2024-01-01', periods=days, freq='B') # B 代表工作日

# 生成随机收益率 (正态分布)
# loc=均值, scale=标准差
daily_returns = np.random.normal(loc=mu, scale=sigma, size=days)

# 计算股价路径 (起始价 * (1+r) 的累积乘积)
price_path = start_price * (1 + daily_returns).cumprod()

# 封装进 DataFrame，保持和之前代码兼容的结构
data = pd.DataFrame(data={'Close': price_path}, index=dates)

print(f"✅ 模拟完成！生成了 {len(data)} 个交易日的数据。")


# === 2. 计算逻辑 (保持不变) ===
# 计算对数收益率
data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))

# 计算 30 天滚动波动率 (标准差) 并年化
data['Volatility'] = data['Log_Return'].rolling(window=30).std() * np.sqrt(252)


# === 3. 绘图 (保持不变) ===
plt.figure(figsize=(12, 10))

# 上图：模拟股价
plt.subplot(2, 1, 1)
plt.plot(data.index, data['Close'], label='Synthetic Price')
plt.title('Synthetic TSLA-like Price Analysis')
plt.legend()
plt.grid(True)

# 下图：年化波动率
plt.subplot(2, 1, 2)
plt.plot(data.index, data['Volatility'], label='30-Day Annualized Volatility (Log Returns)', color='orange')

# 添加平均线
avg_vol = data['Volatility'].mean()
plt.axhline(avg_vol, color='r', linestyle='--', label=f'Average Vol ({avg_vol:.2f})')

plt.title('Volatility Structure (Simulated)')
plt.legend()
plt.grid(True)

plt.tight_layout()
print("📊 绘图完成！窗口已弹出。")
plt.show()
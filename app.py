import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(page_title="黄金销售量预测工具", page_icon="📈", layout="wide")

# 标题
st.title("📈 黄金销售量智能预测工具")
st.markdown("上传历史销售数据，AI 自动预测未来趋势")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 预测设置")
    forecast_days = st.slider("预测天数", 7, 90, 30)
    train_ratio = st.slider("训练数据比例", 0.5, 0.9, 0.8)
    
    st.markdown("---")
    st.markdown("**数据格式要求：**")
    st.markdown("- 必须包含 '交易日期' 列")
    st.markdown("- 必须包含 '销售量（克）' 列")
    st.markdown("- 支持 .xlsx 或 .xls 格式")

# 文件上传
uploaded_file = st.file_uploader("📁 上传销售数据", type=['xlsx', 'xls'])

if uploaded_file is not None:
    try:
        # 读取数据
        df = pd.read_excel(uploaded_file)
        
        # 检查必要列
        required_cols = ['交易日期', '销售量（克）']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ 缺少必要列：{missing_cols}")
            st.info("请确保 Excel 中包含 '交易日期' 和 '销售量（克）' 两列")
        else:
            # 数据预处理
            df['交易日期'] = pd.to_datetime(df['交易日期'])
            df = df.groupby('交易日期')['销售量（克）'].mean().reset_index()
            df = df.sort_values('交易日期').reset_index(drop=True)
            
            # 数据展示
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("数据天数", len(df))
            with col2:
                st.metric("平均日销量", f"{df['销售量（克）'].mean():.2f}克")
            with col3:
                st.metric("最高日销量", f"{df['销售量（克）'].max():.2f}克")
            
            # 原始数据图表
            st.subheader("📊 历史销售趋势")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df['交易日期'], df['销售量（克）'], marker='o', linewidth=1, markersize=3)
            ax.set_xlabel('日期')
            ax.set_ylabel('销售量（克）')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # 预测按钮
            if st.button("🚀 开始 AI 预测", type="primary"):
                with st.spinner('AI 分析中，请稍候...'):
                    
                    # 划分训练集和测试集
                    train_size = int(len(df) * train_ratio)
                    train_data = df['销售量（克）'][:train_size]
                    test_data = df['销售量（克）'][train_size:]
                    
                    # 训练 ARIMA 模型
                    try:
                        model = ARIMA(train_data, order=(5, 1, 0))
                        model_fit = model.fit()
                        
                        # 预测测试集（用于评估）
                        test_predict = model_fit.forecast(steps=len(test_data))
                        
                        # 评估指标
                        mse = mean_squared_error(test_data, test_predict)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(test_data, test_predict)
                        r2 = r2_score(test_data, test_predict)
                        
                        # 用全部数据重新训练，预测未来
                        final_model = ARIMA(df['销售量（克）'], order=(5, 1, 0))
                        final_model_fit = final_model.fit()
                        future_forecast = final_model_fit.forecast(steps=forecast_days)
                        
                        # 生成未来日期
                        last_date = df['交易日期'].iloc[-1]
                        future_dates = pd.date_range(start=last_date, periods=forecast_days+1, freq='B')[1:]
                        
                        # 评估指标展示
                        st.subheader("📋 模型评估")
                        eval_col1, eval_col2, eval_col3, eval_col4 = st.columns(4)
                        with eval_col1:
                            st.metric("RMSE", f"{rmse:.2f}")
                        with eval_col2:
                            st.metric("MAE", f"{mae:.2f}")
                        with eval_col3:
                            st.metric("R² Score", f"{r2:.2f}")
                        with eval_col4:
                            st.metric("预测天数", f"{forecast_days}天")
                        
                        # 预测结果图表
                        st.subheader("🔮 预测结果")
                        fig2, ax2 = plt.subplots(figsize=(14, 6))
                        
                        # 历史数据
                        ax2.plot(df['交易日期'], df['销售量（克）'], 
                                label='历史数据', color='blue', linewidth=1.5)
                        
                        # 测试集预测（如果有）
                        if len(test_data) > 0:
                            test_dates = df['交易日期'][train_size:]
                            ax2.plot(test_dates, test_predict, 
                                    label='测试集预测', color='green', linestyle='--', alpha=0.7)
                        
                        # 未来预测
                        ax2.plot(future_dates, future_forecast, 
                                label='未来预测', color='red', linewidth=2, marker='o', markersize=4)
                        
                        ax2.axvline(x=last_date, color='gray', linestyle=':', alpha=0.5, label='预测起点')
                        ax2.set_xlabel('日期')
                        ax2.set_ylabel('销售量（克）')
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)
                        st.pyplot(fig2)
                        
                        # 预测数据表格
                        st.subheader("📄 详细预测数据")
                        forecast_df = pd.DataFrame({
                            '日期': future_dates,
                            '预测销售量（克）': future_forecast.round(2),
                            '预测区间下限': (future_forecast * 0.9).round(2),
                            '预测区间上限': (future_forecast * 1.1).round(2)
                        })
                        st.dataframe(forecast_df, use_container_width=True)
                        
                        # 下载按钮
                        csv = forecast_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="⬇️ 下载完整预测报告 (CSV)",
                            data=csv,
                            file_name=f"黄金销售预测_{forecast_days}天.csv",
                            mime="text/csv"
                        )
                        
                        # 分析建议
                        st.subheader("💡 智能分析建议")
                        avg_forecast = future_forecast.mean()
                        last_avg = df['销售量（克）'].tail(30).mean()
                        trend = "上升" if avg_forecast > last_avg else "下降"
                        
                        st.info(f"""
                        - 未来{forecast_days}天平均日销量预测：**{avg_forecast:.2f}克**
                        - 与近30天平均（{last_avg:.2f}克）相比呈**{trend}趋势**
                        - 建议根据预测提前调整库存和采购计划
                        """)
                        
                    except Exception as e:
                        st.error(f"预测出错：{str(e)}")
                        st.info("请检查数据是否足够（建议至少60天数据）")
                        
    except Exception as e:
        st.error(f"文件读取失败：{str(e)}")
        st.info("请确保上传的是有效的 Excel 文件")

else:
    # 示例展示
    st.info("👆 请上传数据文件开始分析")
    
    with st.expander("📝 查看示例数据格式"):
        sample_data = pd.DataFrame({
            '交易日期': ['2024-01-01', '2024-01-02', '2024-01-03'],
            '销售量（克）': [150.5, 180.2, 165.8]
        })
        st.write(sample_data)
        st.download_button(
            "下载示例模板",
            sample_data.to_csv(index=False),
            "示例数据模板.csv"
        )

# 页脚
st.markdown("---")
st.caption("技术支持 | 基于 ARIMA 时序预测模型")

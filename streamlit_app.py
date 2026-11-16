import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------
# KONFIGURASI HALAMAN
# ---------------------------------------------------------
st.set_page_config(page_title="SMC & FVG Rejection Screener", layout="wide")
st.title("📊 SMC & FVG Re-Entry + Valid Rejection Screener")
st.markdown("Deteksi struktur pasar (**BOS/CHoCH**), **Order Block**, **Fair Value Gap (FVG)**, serta validasi **Price Rejection** otomatis.")

# ---------------------------------------------------------
# SIDEBAR INPUT
# ---------------------------------------------------------
st.sidebar.header("⚙️ Pengaturan Screener")
ticker = st.sidebar.text_input("Simbol / Ticker (Contoh: GC=F, BBCA.JK, KIJA.JK)", value="GC=F")
timeframe = st.sidebar.selectbox("Timeframe", ["15m", "30m", "1h", "4h", "1d"], index=2)
period = st.sidebar.selectbox("Periode Data", ["5d", "1mo", "3mo", "6mo", "1y"], index=1)
swing_length = st.sidebar.slider("Swing Lookback Length", min_value=5, max_value=50, value=15)
fvg_auto_threshold = st.sidebar.checkbox("Aktifkan Auto Threshold FVG", value=True)

# ---------------------------------------------------------
# FUNGSI AMBIL DATA
# ---------------------------------------------------------
@st.cache_data(ttl=60)
def load_data(symbol, tf, per):
    df = yf.download(symbol, interval=tf, period=per, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df

# ---------------------------------------------------------
# LOGIKA SMC, FVG & VALID REJECTION
# ---------------------------------------------------------
def analyze_smc(df, swing_len=15, use_fvg_thresh=True):
    df = df.copy()
    
    # 1. Hitung ATR untuk filter volatilitas
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1)))
    )
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # 2. Deteksi Swing High & Swing Low
    df['Swing_High'] = df['High'][(df['High'] == df['High'].rolling(swing_len*2+1, center=True).max())]
    df['Swing_Low'] = df['Low'][(df['Low'] == df['Low'].rolling(swing_len*2+1, center=True).min())]

    last_swing_high = df['Swing_High'].dropna().iloc[-1] if not df['Swing_High'].dropna().empty else df['High'].max()
    last_swing_low = df['Swing_Low'].dropna().iloc[-1] if not df['Swing_Low'].dropna().empty else df['Low'].min()
    
    # 3. Zona Premium, Equilibrium, Discount
    equilibrium = (last_swing_high + last_swing_low) / 2
    
    # 4. Penentuan Bias Tren Sederhana (BOS / CHoCH)
    current_close = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    
    # Cek apakah harga baru saja menembus swing terakhir
    trend_bias = "BULLISH" if current_close > equilibrium else "BEARISH"
    
    # 5. Deteksi Fair Value Gaps (FVG) yang belum termitigasi penuh
    fvgs = []
    for i in range(2, len(df)):
        # Bullish FVG: Low lilin ke-i > High lilin ke-(i-2)
        if df['Low'].iloc[i] > df['High'].iloc[i-2] and df['Close'].iloc[i-1] > df['High'].iloc[i-2]:
            gap_size = df['Low'].iloc[i] - df['High'].iloc[i-2]
            atr_val = df['ATR'].iloc[i] if not np.isnan(df['ATR'].iloc[i]) else 0
            if not use_fvg_thresh or (gap_size > atr_val * 0.2):
                # Cek apakah FVG sudah ditembus ke bawah setelah terbentuk
                future_lows = df['Low'].iloc[i+1:]
                if future_lows.empty or future_lows.min() > df['High'].iloc[i-2]:
                    fvgs.append({
                        'type': 'BULLISH_FVG',
                        'top': df['Low'].iloc[i],
                        'bottom': df['High'].iloc[i-2],
                        'time': df.index[i-1]
                    })
                    
        # Bearish FVG: High lilin ke-i < Low lilin ke-(i-2)
        elif df['High'].iloc[i] < df['Low'].iloc[i-2] and df['Close'].iloc[i-1] < df['Low'].iloc[i-2]:
            gap_size = df['Low'].iloc[i-2] - df['High'].iloc[i]
            atr_val = df['ATR'].iloc[i] if not np.isnan(df['ATR'].iloc[i]) else 0
            if not use_fvg_thresh or (gap_size > atr_val * 0.2):
                future_highs = df['High'].iloc[i+1:]
                if future_highs.empty or future_highs.max() < df['Low'].iloc[i-2]:
                    fvgs.append({
                        'type': 'BEARISH_FVG',
                        'top': df['Low'].iloc[i-2],
                        'bottom': df['High'].iloc[i],
                        'time': df.index[i-1]
                    })

    # Ambil 3 FVG aktif terbaru
    active_fvgs = fvgs[-3:] if len(fvgs) > 0 else []

    # 6. Deteksi Valid Rejection pada Lilin Terakhir / Sebelumnya
    rejection_signals = []
    for idx in [-1, -2]:  # Cek 2 lilin terakhir
        row = df.iloc[idx]
        candle_time = df.index[idx]
        body = abs(row['Close'] - row['Open'])
        total_range = row['High'] - row['Low']
        if total_range == 0:
            continue
            
        lower_wick = min(row['Open'], row['Close']) - row['Low']
        upper_wick = row['High'] - max(row['Open'], row['Close'])

        # Cek interaksi dengan Active FVG
        for fvg in active_fvgs:
            # VALID BULLISH REJECTION:
            # Harga masuk zona Bullish FVG, berada di area Discount (< Equilibrium),
            # sumbu bawah >= 45% dari total candle, dan close di atas batas bawah FVG
            if fvg['type'] == 'BULLISH_FVG':
                touched_fvg = (row['Low'] <= fvg['top']) and (row['Close'] >= fvg['bottom'])
                in_discount = row['Low'] <= equilibrium
                valid_wick = (lower_wick / total_range) >= 0.45 and row['Close'] > row['Open']
                
                if touched_fvg and in_discount and valid_wick:
                    rejection_signals.append({
                        'time': candle_time,
                        'type': 'VALID BULLISH REJECTION (BUY)',
                        'price': row['Low'],
                        'entry_zone': f"{fvg['bottom']:.2f} - {fvg['top']:.2f}",
                        'sl': f"{fvg['bottom'] - (df['ATR'].iloc[idx]*0.5):.2f}"
                    })

            # VALID BEARISH REJECTION:
            # Harga masuk zona Bearish FVG, berada di area Premium (> Equilibrium),
            # sumbu atas >= 45% dari total candle, dan close di bawah batas atas FVG
            elif fvg['type'] == 'BEARISH_FVG':
                touched_fvg = (row['High'] >= fvg['bottom']) and (row['Close'] <= fvg['top'])
                in_premium = row['High'] >= equilibrium
                valid_wick = (upper_wick / total_range) >= 0.45 and row['Close'] < row['Open']
                
                if touched_fvg and in_premium and valid_wick:
                    rejection_signals.append({
                        'time': candle_time,
                        'type': 'VALID BEARISH REJECTION (SELL)',
                        'price': row['High'],
                        'entry_zone': f"{fvg['bottom']:.2f} - {fvg['top']:.2f}",
                        'sl': f"{fvg['top'] + (df['ATR'].iloc[idx]*0.5):.2f}"
                    })

    return df, last_swing_high, last_swing_low, equilibrium, active_fvgs, rejection_signals

# ---------------------------------------------------------
# EKSEKUSI & VISUALISASI
# ---------------------------------------------------------
if ticker:
    data = load_data(ticker, timeframe, period)
    
    if data.empty or len(data) < 30:
        st.error("Data tidak ditemukan atau terlalu sedikit. Coba ganti ticker atau perbesar periode.")
    else:
        df, sh, sl, eq, active_fvgs, rejections = analyze_smc(data, swing_length, fvg_auto_threshold)
        last_price = df['Close'].iloc[-1]

        # METRIK RINGKASAN
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Harga Terakhir", f"{last_price:,.2f}")
        col2.metric("Swing High (Premium)", f"{sh:,.2f}")
        col3.metric("Equilibrium (50%)", f"{eq:,.2f}")
        col4.metric("Swing Low (Discount)", f"{sl:,.2f}")

        # STATUS REJECTION / ALERT SINYAL
        st.subheader("🚨 Status Sinyal & Valid Rejection")
        if rejections:
            for sig in rejections:
                if "BUY" in sig['type']:
                    st.success(f"**{sig['type']}** terdeteksi pada `{sig['time']}` | Area Re-Entry FVG: **{sig['entry_zone']}** | Saran SL: **{sig['sl']}**")
                else:
                    st.error(f"**{sig['type']}** terdeteksi pada `{sig['time']}` | Area Re-Entry FVG: **{sig['entry_zone']}** | Saran SL: **{sig['sl']}**")
        else:
            zone_status = "DISCOUNT ZONE (Cari Peluang Buy)" if last_price < eq else "PREMIUM ZONE (Cari Peluang Sell)"
            st.info(f"Belum ada candle rejection valid pada 2 bar terakhir. Posisi harga saat ini berada di **{zone_status}**.")

        # TABEL ZONA FVG AKTIF (POINT RE-ENTRY)
        if active_fvgs:
            st.subheader("🎯 Daftar Zona Re-Entry Aktif (Unmitigated FVG)")
            fvg_df = pd.DataFrame(active_fvgs)
            fvg_df['Status Zona'] = np.where(fvg_df['top'] < eq, 'Discount (Ideal Buy)', 'Premium (Ideal Sell)')
            st.dataframe(fvg_df[['time', 'type', 'bottom', 'top', 'Status Zona']], use_container_width=True)

        # PLOT CHART INTERAKTIF DENGAN PLOTLY
        fig = go.Figure()

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price"
        ))

        # Garis Swing High, Equilibrium, Swing Low
        fig.add_hline(y=sh, line_dash="dash", line_color="red", annotation_text="Swing High (Premium)")
        fig.add_hline(y=eq, line_dash="dot", line_color="gray", annotation_text="Equilibrium (50%)")
        fig.add_hline(y=sl, line_dash="dash", line_color="green", annotation_text="Swing Low (Discount)")

        # Gambar Kotak FVG Aktif
        for fvg in active_fvgs:
            color = "rgba(0, 255, 104, 0.25)" if fvg['type'] == 'BULLISH_FVG' else "rgba(255, 0, 8, 0.25)"
            fig.add_shape(
                type="rect",
                x0=fvg['time'],
                x1=df.index[-1],
                y0=fvg['bottom'],
                y1=fvg['top'],
                fillcolor=color,
                line=dict(width=0),
                layer="below"
            )

        # Tandai Valid Rejection di Chart
        for sig in rejections:
            is_buy = "BUY" in sig['type']
            fig.add_trace(go.Scatter(
                x=[sig['time']],
                y=[sig['price']],
                mode="markers+text",
                marker=dict(
                    symbol="triangle-up" if is_buy else "triangle-down",
                    size=16,
                    color="lime" if is_buy else "red"
                ),
                text=["VALID REJECT BUY" if is_buy else "VALID REJECT SELL"],
                textposition="bottom center" if is_buy else "top center",
                name=sig['type']
            ))

        fig.update_layout(
            title=f"Chart SMC & FVG Rejection - {ticker} ({timeframe})",
            xaxis_rangeslider_visible=False,
            height=650,
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

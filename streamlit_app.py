import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="XAUUSD SMC & FVG Visual Screener", page_icon="📈", layout="wide"
)

st.title("🎯 XAUUSD Smart Money Concepts (SMC) & FVG Screener")
st.markdown(
    "Aplikasi visual interaktif untuk mendeteksi **Fair Value Gap (FVG)**, **Displacement**, dan struktur harga institusional pada XAUUSD."
)

# Sidebar Pengaturan
st.sidebar.header("⚙️ Pengaturan Market & TF")
timeframe = st.sidebar.selectbox(
    "Pilih Timeframe", ["15m", "1h", "4h", "1d"], index=1
)
period = st.sidebar.selectbox(
    "Rentang Histori Data", ["5d", "1mo", "3mo", "6mo"], index=1
)


# Tarik Data dari YFinance
@st.cache_data(ttl=300)
def load_gold_data(period, interval):
    df = yf.download("GC=F", period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df


data = load_gold_data(period, timeframe)

if data.empty:
    st.error(
        "Gagal memuat data pasar. Silakan coba ubah timeframe atau rentang waktu."
    )
else:
    # Metrik Harga Terkini
    latest_close = data["Close"].iloc[-1]
    prev_close = data["Close"].iloc[-2]
    change = latest_close - prev_close
    pct_change = (change / prev_close) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Harga XAUUSD Terkini",
        f"${latest_close:,.2f}",
        f"{change:+.2f} ({pct_change:+.2f}%)",
    )
    col2.metric("Timeframe Aktif", timeframe.upper())
    col3.metric("Total Candle Dianalisis", len(data))

    st.markdown("---")


    # -------------------------------------------------------------
    # LOGIKA DETEKSI FVG & KOTAK ZONA
    # -------------------------------------------------------------
    def detect_fvg(df):
        zones = []
        for i in range(1, len(df) - 1):
            c_prev = df.iloc[i - 1]
            c_curr = df.iloc[i]  # Candle impulsif
            c_next = df.iloc[i + 1]

            body_size = abs(c_curr["Close"] - c_curr["Open"])
            avg_body = abs(df["Close"] - df["Open"]).mean()
            is_displacement = body_size > (1.3 * avg_body)

            # Bullish FVG (Demand Zone)
            if (
                c_next["Low"] > c_prev["High"]
                and c_curr["Close"] > c_curr["Open"]
            ):
                zones.append(
                    {
                        "Time": df.index[i],
                        "Type": "🟢 Bullish FVG (Demand)",
                        "Zone Low": c_prev["High"],
                        "Zone High": c_next["Low"],
                        "Displacement": is_displacement,
                    }
                )
            # Bearish FVG (Supply Zone)
            elif (
                c_next["High"] < c_prev["Low"]
                and c_curr["Close"] < c_curr["Open"]
            ):
                zones.append(
                    {
                        "Time": df.index[i],
                        "Type": "🔴 Bearish FVG (Supply)",
                        "Zone High": c_prev["Low"],
                        "Zone Low": c_next["High"],
                        "Displacement": is_displacement,
                    }
                )
        return pd.DataFrame(zones)


    fvg_df = detect_fvg(data)

    # -------------------------------------------------------------
    # VISUALISASI CANDLESTICK + KOTAK ZONA (PLOTLY)
    # -------------------------------------------------------------
    st.subheader("📊 Visualisasi Chart & Zona Kotak FVG")

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="XAUUSD",
            )
        ]
    )

    # Tambahkan kotak zona FVG ke chart jika ada
    if not fvg_df.empty:
        recent_zones = fvg_df.tail(8)
        for _, row in recent_zones.iterrows():
            color = (
                "rgba(0, 255, 0, 0.2)"
                if "Bullish" in row["Type"]
                else "rgba(255, 0, 0, 0.2)"
            )
            line_color = "green" if "Bullish" in row["Type"] else "red"

            fig.add_shape(
                type="rect",
                x0=row["Time"],
                y0=row["Zone Low"],
                x1=data.index[-1],
                y1=row["Zone High"],
                fillcolor=color,
                line=dict(color=line_color, width=1, dash="dot"),
            )

    fig.update_layout(
        title="XAUUSD Price Action & Institutional FVG Zones",
        yaxis_title="Harga (USD)",
        xaxis_title="Waktu",
        template="plotly_dark",
        height=550,
        xaxis_rangeslider_visible=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------
    # TABEL DETAIL ZONA
    # -------------------------------------------------------------
    st.subheader("📋 Daftar Kotak Zona Aktif")
    if not fvg_df.empty:
        st.dataframe(
            fvg_df.tail(10).iloc[::-1][
                ["Time", "Type", "Zone Low", "Zone High", "Displacement"]
            ],
            use_container_width=True,
        )
        st.info(
            "💡 **Petunjuk Penggunaan:** Kotak transparan di chart menunjukkan area *Fair Value Gap (FVG)*. "
            "Tunggu harga melakukan retrace (koreksi) kembali ke dalam kotak zona tersebut untuk mencari peluang reaksi harga."
        )
    else:
        st.warning("Belum ada zona FVG signifikan yang terbentuk pada rentang ini.")

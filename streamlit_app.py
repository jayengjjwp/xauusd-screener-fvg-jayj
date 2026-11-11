from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="XAUUSD SMC & FVG Live Screener", page_icon="📈", layout="wide"
)

st.title("🎯 XAUUSD Live SMC & FVG Screener")
st.markdown(
    f"📅 **Waktu Analisis Terkini:** {datetime.now().strftime('%A, %d %B %Y - %H:%M')} WIB | "
    "Screener otomatis zona *Fair Value Gap (FVG)* dan rekomendasi sinyal aksi."
)

# Sidebar Pengaturan
st.sidebar.header("⚙️ Pengaturan Market & TF")
timeframe = st.sidebar.selectbox(
    "Pilih Timeframe", ["15m", "1h", "4h", "1d"], index=1
)
period = st.sidebar.selectbox(
    "Rentang Histori Data", ["5d", "1mo", "3mo", "6mo"], index=1
)


# Tarik Data dari YFinance (Tanpa Cache Lama agar Selalu Fresh)
@st.cache_data(ttl=60)
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
    # LOGIKA DETEKSI FVG & PENENTUAN SINYAL (BUY / SELL)
    # -------------------------------------------------------------
    def detect_fvg_and_signals(df):
        zones = []
        for i in range(1, len(df) - 1):
            c_prev = df.iloc[i - 1]
            c_curr = df.iloc[i]  # Candle impulsif
            c_next = df.iloc[i + 1]

            body_size = abs(c_curr["Close"] - c_curr["Open"])
            avg_body = abs(df["Close"] - df["Open"]).mean()
            is_displacement = body_size > (1.2 * avg_body)

            # Bullish FVG (Demand Zone) -> Potensi BUY
            if (
                c_next["Low"] > c_prev["High"]
                and c_curr["Close"] > c_curr["Open"]
            ):
                zones.append(
                    {
                        "Time": df.index[i],
                        "Type": "🟢 Bullish FVG (Demand)",
                        "Action": "BUY (Tunggu Retrace ke Zona)",
                        "Zone Low": c_prev["High"],
                        "Zone High": c_next["Low"],
                        "Displacement": is_displacement,
                    }
                )
            # Bearish FVG (Supply Zone) -> Potensi SELL
            elif (
                c_next["High"] < c_prev["Low"]
                and c_curr["Close"] < c_curr["Open"]
            ):
                zones.append(
                    {
                        "Time": df.index[i],
                        "Type": "🔴 Bearish FVG (Supply)",
                        "Action": "SELL (Tunggu Retrace ke Zona)",
                        "Zone Low": c_next["High"],
                        "Zone High": c_prev["Low"],
                        "Displacement": is_displacement,
                    }
                )
        return pd.DataFrame(zones)


    fvg_df = detect_fvg_and_signals(data)

    # Status Sinyal Berdasarkan Zona Terakhir
    st.subheader("🚨 Rekomendasi Sinyal Aksi Terkini")
    if not fvg_df.empty:
        latest_signal = fvg_df.iloc[-1]

        if "Bullish" in latest_signal["Type"]:
            st.success(
                f"**REKOMENDASI AKSI: 🟢 BUY SETUP**\n\n"
                f"- **Alasan:** Terdeteksi *Bullish FVG* (Demand Zone) terbaru pada {latest_signal['Time']}.\n"
                f"- **Zona Area Beli:** \$ {latest_signal['Zone Low']:,.2f} - \$ {latest_signal['Zone High']:,.2f}\n"
                f"- **Saran Eksekusi:** Tunggu harga turun (retrace) masuk ke dalam kotak zona tersebut sebelum melakukan *Buy*."
            )
        else:
            st.error(
                f"**REKOMENDASI AKSI: 🔴 SELL SETUP**\n\n"
                f"- **Alasan:** Terdeteksi *Bearish FVG* (Supply Zone) terbaru pada {latest_signal['Time']}.\n"
                f"- **Zona Area Jual:** \$ {latest_signal['Zone Low']:,.2f} - \$ {latest_signal['Zone High']:,.2f}\n"
                f"- **Saran Eksekusi:** Tunggu harga naik (retrace) masuk ke dalam kotak zona tersebut sebelum melakukan *Sell*."
            )
    else:
        st.info(
            "⏳ Belum ada sinyal aksi tegas (pasar sedang konsolidasi atau menunggu displacement baru)."
        )

    st.markdown("---")

    # -------------------------------------------------------------
    # VISUALISASI CHART & KOTAK ZONA (PLOTLY)
    # -------------------------------------------------------------
    st.subheader("📊 Visualisasi Chart & Kotak Zona Institusional")

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
        title="XAUUSD Price Action & FVG Action Zones",
        yaxis_title="Harga (USD)",
        xaxis_title="Waktu",
        template="plotly_dark",
        height=550,
        xaxis_rangeslider_visible=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------
    # TABEL DETAIL ZONA & AKSI
    # -------------------------------------------------------------
    st.subheader("📋 Tabel Riwayat Zona & Sinyal Aksi")
    if not fvg_df.empty:
        st.dataframe(
            fvg_df.tail(10).iloc[::-1][
                ["Time", "Type", "Action", "Zone Low", "Zone High"]
            ],
            use_container_width=True,
        )

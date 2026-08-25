"""
📊 Borsa Analiz Uygulaması (Ücretsiz Sürüm - Google Gemini)
Kawsar (@Kawsar_Ai) dizisindeki analiz maddelerini baz alır.

Arama çubuğuna bir hisse ticker'ı yaz (örn: AAPL, MSFT, THYAO.IS, ASELS.IS)
Uygulama gerçek finansal verileri çeker ve 6 kritere göre analiz üretir.

Analizler için ÜCRETSİZ Google Gemini API anahtarı kullanılır:
    https://aistudio.google.com/apikey  (kredi kartı gerekmez)

Çalıştırmak için:
    pip install -r requirements.txt
    streamlit run app.py
"""

import time
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import streamlit as st
import yfinance as yf
from streamlit_searchbox import st_searchbox
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Sayfa ayarları
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Borsa Analiz Uygulaması",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# GRAFİK MODELLERİ REHBERİ — şematik çizimler (matplotlib ile üretilir)
# Her desen küçük bir çizim + açıklama + nasıl yorumlanır bilgisiyle gösterilir.
# ---------------------------------------------------------------------------
def _fig(draw_fn, bullish=None):
    fig, ax = plt.subplots(figsize=(3.2, 2.0), dpi=110)
    draw_fn(ax)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.margins(0.05)
    fig.tight_layout(pad=0.3)
    return fig


def _line(ax, xs, ys, color="#1f77b4"):
    ax.plot(xs, ys, color=color, linewidth=1.8)


def d_head_shoulders(ax):
    x = np.arange(11)
    y = [1, 2, 1.5, 3, 1.6, 4.2, 1.6, 3, 1.5, 2, 1]
    _line(ax, x, y, "#d62728")
    ax.axhline(1.55, ls="--", color="gray", lw=1)  # boyun çizgisi


def d_inv_head_shoulders(ax):
    x = np.arange(11)
    y = [5, 4, 4.5, 3, 4.4, 1.8, 4.4, 3, 4.5, 4, 5]
    _line(ax, x, y, "#2ca02c")
    ax.axhline(4.45, ls="--", color="gray", lw=1)


def d_double_top(ax):
    x = np.arange(9)
    y = [1, 2.5, 4, 2.6, 4, 2.5, 3, 1.5, 0.8]
    _line(ax, x, y, "#d62728")
    ax.axhline(2.55, ls="--", color="gray", lw=1)


def d_double_bottom(ax):
    x = np.arange(9)
    y = [5, 3.5, 2, 3.4, 2, 3.5, 3, 4.5, 5.2]
    _line(ax, x, y, "#2ca02c")
    ax.axhline(3.45, ls="--", color="gray", lw=1)


def d_triple_top(ax):
    x = np.arange(11)
    y = [1, 3, 4, 3, 4, 3, 4, 3, 2, 1.2, 0.8]
    _line(ax, x, y, "#d62728")
    ax.axhline(3, ls="--", color="gray", lw=1)


def d_rising_wedge(ax):
    x = np.arange(10)
    hi = 2 + 0.35 * x
    lo = 1 + 0.5 * x
    y = [lo[i] if i % 2 else hi[i] for i in range(10)]
    _line(ax, x, y, "#d62728")
    _line(ax, x, hi, "gray"); _line(ax, x, lo, "gray")


def d_falling_wedge(ax):
    x = np.arange(10)
    hi = 5 - 0.5 * x
    lo = 4 - 0.35 * x
    y = [hi[i] if i % 2 else lo[i] for i in range(10)]
    _line(ax, x, y, "#2ca02c")
    _line(ax, x, hi, "gray"); _line(ax, x, lo, "gray")


def d_cup_handle(ax):
    x1 = np.linspace(0, 6, 40)
    cup = 3 + 1.5 * ((x1 - 3) / 3) ** 2 - 1.5
    x2 = np.linspace(6, 8, 15)
    handle = 3 - 0.4 * np.sin((x2 - 6) * np.pi / 2)
    x3 = np.linspace(8, 9, 8)
    breakout = 3 + (x3 - 8) * 1.2
    _line(ax, np.r_[x1, x2, x3], np.r_[cup, handle, breakout], "#2ca02c")


def d_flag(ax):
    x1 = np.linspace(0, 3, 10); pole = 1 + x1 * 1.2
    x2 = np.linspace(3, 6, 10); flag = 4.6 - (x2 - 3) * 0.25
    x3 = np.linspace(6, 8, 8); out = flag[-1] + (x3 - 6) * 1.1
    _line(ax, np.r_[x1, x2, x3], np.r_[pole, flag, out], "#2ca02c")


def d_pennant(ax):
    x1 = np.linspace(0, 3, 10); pole = 1 + x1 * 1.2
    x2 = np.arange(3, 8)
    hi = [4.6, 4.3, 4.0, 3.8, 3.7]; lo = [4.6, 3.2, 3.4, 3.55, 3.65]
    seq = [hi[i] if i % 2 == 0 else lo[i] for i in range(5)]
    x3 = np.linspace(8, 9.5, 6); out = 3.7 + (x3 - 8) * 1.1
    _line(ax, np.r_[x1, x2, x3], np.r_[pole, seq, out], "#2ca02c")


def d_sym_triangle(ax):
    x = np.arange(9)
    hi = 5 - 0.3 * x; lo = 1 + 0.3 * x
    y = [hi[i] if i % 2 == 0 else lo[i] for i in range(9)]
    _line(ax, x, y, "#1f77b4")
    _line(ax, x, hi, "gray"); _line(ax, x, lo, "gray")


def d_asc_triangle(ax):
    x = np.arange(9)
    hi = np.full(9, 4.0); lo = 1 + 0.35 * x
    y = [hi[i] if i % 2 == 0 else lo[i] for i in range(9)]
    _line(ax, x, y, "#2ca02c")
    _line(ax, x, hi, "gray"); _line(ax, x, lo, "gray")


def d_desc_triangle(ax):
    x = np.arange(9)
    lo = np.full(9, 1.5); hi = 5 - 0.35 * x
    y = [lo[i] if i % 2 == 0 else hi[i] for i in range(9)]
    _line(ax, x, y, "#d62728")
    _line(ax, x, hi, "gray"); _line(ax, x, lo, "gray")


def d_rectangle(ax):
    x = np.arange(9)
    y = [1.5, 4, 1.5, 4, 1.5, 4, 1.5, 4, 5.2]
    _line(ax, x, y, "#1f77b4")
    ax.axhline(4, ls="--", color="gray", lw=1); ax.axhline(1.5, ls="--", color="gray", lw=1)


def _candle(ax, x, o, c, h, l, w=0.3):
    color = "#2ca02c" if c >= o else "#d62728"
    ax.plot([x, x], [l, h], color="black", lw=1)
    ax.add_patch(plt.Rectangle((x - w, min(o, c)), 2 * w, abs(c - o) or 0.05,
                               color=color))


def d_doji(ax):
    for i, (o, c, h, l) in enumerate([(2, 2.5, 3, 1.5), (3, 3.1, 3.2, 2.9), (2.8, 2.3, 3.3, 1.8)]):
        _candle(ax, i, o, c, h, l)
    ax.set_xlim(-1, 3); ax.set_ylim(1, 4)


def d_hammer(ax):
    _candle(ax, 0, 3.5, 3.0, 3.6, 3.4)
    _candle(ax, 1, 2.6, 2.9, 3.0, 1.2)  # uzun alt fitil
    _candle(ax, 2, 2.9, 3.6, 3.7, 2.8)
    ax.set_xlim(-1, 3); ax.set_ylim(1, 4)


def d_engulfing(ax):
    _candle(ax, 0, 3.4, 2.9, 3.5, 2.8)  # küçük düşüş
    _candle(ax, 1, 2.8, 3.8, 3.9, 2.7)  # büyük yükseliş yutan
    ax.set_xlim(-1, 2); ax.set_ylim(2, 4)


def d_star(ax):
    _candle(ax, 0, 4, 2.8, 4.1, 2.7)   # büyük düşüş
    _candle(ax, 1, 2.5, 2.4, 2.7, 2.3) # küçük yıldız
    _candle(ax, 2, 2.6, 3.9, 4.0, 2.5) # büyük yükseliş
    ax.set_xlim(-1, 3); ax.set_ylim(2, 4.2)


PATTERNS = {
    "Dönüş (Reversal) Desenleri": [
        (d_head_shoulders, "Omuz-Baş-Omuz", "Yükseliş trendinin sonu. Üç tepe: ortadaki (baş) en yüksek. "
         "Boyun çizgisi (kesikli) aşağı kırılırsa düşüşe dönüş sinyali."),
        (d_inv_head_shoulders, "Ters Omuz-Baş-Omuz", "Düşüş trendinin sonu; dip sinyali. "
         "Boyun çizgisi yukarı kırılırsa yükselişe dönüş."),
        (d_double_top, "Çift Tepe (M)", "Fiyat iki kez aynı seviyeye çıkıp geri döner. "
         "Ortadaki dip kırılırsa düşüş sinyali. En güvenilir dönüş desenlerinden."),
        (d_double_bottom, "Çift Dip (W)", "İki kez aynı dibe inip yükselir. "
         "Ortadaki tepe kırılırsa yükseliş sinyali."),
        (d_triple_top, "Üçlü Tepe/Dip", "Çift versiyonun daha güçlü hali; üç kez denenen seviye."),
        (d_rising_wedge, "Yükselen Kama", "Yükselirken daralır. Genelde AŞAĞI kırılır → düşüş sinyali."),
        (d_falling_wedge, "Alçalan Kama", "Düşerken daralır. Genelde YUKARI kırılır → yükseliş sinyali."),
        (d_cup_handle, "Fincan-Kulp", "Uzun 'U' taban + küçük kulp. Kulp kırılınca yükseliş sinyali."),
    ],
    "Devam (Continuation) Desenleri": [
        (d_flag, "Bayrak", "Sert hareket (direk) sonrası ters eğimli kısa mola, sonra aynı yöne devam."),
        (d_pennant, "Flama", "Direk sonrası küçük simetrik üçgen; kırılımla trend devam eder."),
        (d_sym_triangle, "Simetrik Üçgen", "Alım-satım dengelenir, sıkışır. Kırılım yönü trendi belirler."),
        (d_asc_triangle, "Yükselen Üçgen", "Düz direnç + yükselen dipler. Genelde YUKARI kırılır."),
        (d_desc_triangle, "Alçalan Üçgen", "Düz destek + alçalan tepeler. Genelde AŞAĞI kırılır."),
        (d_rectangle, "Dikdörtgen (Kanal)", "Yatay bantta gidip gelme; banttan kırılım yön verir."),
    ],
    "Mum (Candlestick) Formasyonları": [
        (d_doji, "Doji", "Açılış ≈ kapanış. Kararsızlık; trend sonunda dönüş habercisi olabilir."),
        (d_hammer, "Çekiç", "Uzun alt fitil, küçük gövde. Dipte dönüş (yükseliş) sinyali."),
        (d_engulfing, "Yutan Formasyon", "İkinci mum birincinin gövdesini tamamen sarar. Güçlü dönüş sinyali."),
        (d_star, "Sabah/Akşam Yıldızı", "Üç mumluk dönüş: büyük mum + küçük yıldız + ters yönde büyük mum."),
    ],
}


def render_pattern_guide():
    st.title("📚 Grafik Modelleri Rehberi")
    st.caption("Desenleri manuel incelemek için şematik referans. Çizimler temsilîdir.")
    st.info("Bu desenler ihtimal verir, kesinlik değil. Hacim ve göstergelerle (RSI, MACD, "
            "hareketli ortalamalar) birlikte teyit edilmeli.")
    for group, items in PATTERNS.items():
        st.header(group)
        cols = st.columns(2)
        for i, (fn, name, desc) in enumerate(items):
            with cols[i % 2]:
                st.subheader(name)
                st.pyplot(_fig(fn), use_container_width=False)
                st.write(desc)
                st.divider()


# ---------------------------------------------------------------------------
# Analiz kriterleri (Kawsar dizisinden). Yeni madde eklemek için buraya
# bir sözlük eklemen yeterli — arayüz otomatik güncellenir.
# ---------------------------------------------------------------------------
CRITERIA = [
    {
        "key": "moat",
        "title": "🏰 Rekabet Avantajı (Moat) Analizi",
        "prompt": (
            "{company} şirketinin rekabet hendeğini (moat) değerlendir.\n"
            "Şunları tartış:\n"
            "• Marka gücü\n"
            "• Ağ etkileri (network effects)\n"
            "• Geçiş maliyetleri (switching costs)\n"
            "• Maliyet avantajı\n"
            "• Patentler veya tescilli teknoloji\n\n"
            "En güçlü rakipleriyle karşılaştır ve moat'ı 1-10 arasında puanla. "
            "Puanı yanıtın sonunda **Moat Puanı: X/10** formatında ver."
        ),
    },
    {
        "key": "valuation",
        "title": "💰 Hisse Değerlemesi (Yatırım Bankası Gibi)",
        "prompt": (
            "{ticker} hissesi için değerleme analizi yap.\n"
            "Şunları içer:\n"
            "• F/K (P/E) oranı karşılaştırması\n"
            "• İndirgenmiş Nakit Akışı (DCF) tahmini\n"
            "• Sektör ortalaması değerlemesi\n"
            "• Ucuz mu pahalı mı (undervalued/overvalued) sonucu\n\n"
            "Sonucu net olarak **Değerleme Sonucu: Ucuz / Adil / Pahalı** şeklinde belirt."
        ),
    },
    {
        "key": "risk",
        "title": "⚠️ Risk Analizi",
        "prompt": (
            "{company} şirketine yatırım yapmanın en büyük risklerini belirle.\n"
            "Şunları içer:\n"
            "• Ekonomik riskler\n"
            "• Sektörel yıkım (disruption)\n"
            "• Rekabet\n"
            "• Regülasyon tehditleri\n"
            "• Borç veya finansal riskler\n\n"
            "Riskleri en tehlikeliden en aza doğru sırala."
        ),
    },
    {
        "key": "growth",
        "title": "🚀 Büyüme Potansiyeli Analizi",
        "prompt": (
            "{company} şirketinin gelecekteki büyüme potansiyelini analiz et.\n"
            "Şunları değerlendir:\n"
            "• Pazar büyüklüğü\n"
            "• Sektör büyüme oranı\n"
            "• Genişleme fırsatları\n"
            "• Yeni ürünler\n"
            "• Yapay zeka veya teknoloji avantajları\n\n"
            "Önümüzdeki 5-10 yıllık potansiyel büyümeyi tahmin et."
        ),
    },
    {
        "key": "institutional",
        "title": "🏦 Kurumsal Yatırımcı Bakış Açısı",
        "prompt": (
            "Bir hedge fon portföy yöneticisi gibi davran.\n"
            "{ticker} hissesinin uzun vadeli iyi bir yatırım olup olmadığını değerlendir.\n"
            "Şunları içer:\n"
            "• Kurumların neden alabileceği\n"
            "• Neden kaçınabileceği\n"
            "• Kilit katalizörler\n"
            "• Yatırım tezi"
        ),
    },
    {
        "key": "bull_bear",
        "title": "🐂 vs 🐻 Boğa & Ayı Tartışması",
        "prompt": (
            "{ticker} hissesi hakkında iki analist arasında bir tartışma oluştur.\n"
            "Bir analist boğa (bullish), diğeri ayı (bearish).\n"
            "Her biri veriye dayalı argümanlar sunmalı.\n"
            "Dengeli bir sonuçla bitir."
        ),
    },
    {
        "key": "health",
        "title": "🩺 Finansal Sağlık & Bilanço",
        "prompt": (
            "{company} şirketinin finansal sağlığını ve bilanço gücünü değerlendir.\n"
            "Şunları içer:\n"
            "• Likidite ve nakit pozisyonu (cari oran vb.)\n"
            "• Borçluluk düzeyi (Borç/Özkaynak, faiz karşılama)\n"
            "• Serbest nakit akışı üretimi\n"
            "• İflas/mali sıkıntı riski\n\n"
            "Şirketin borcunu çevirebilme ve ayakta kalma gücünü 1-10 arasında puanla. "
            "Puanı **Finansal Sağlık Puanı: X/10** formatında ver."
        ),
    },
    {
        "key": "dividend",
        "title": "💵 Temettü Analizi",
        "prompt": (
            "{company} şirketinin temettü profilini analiz et.\n"
            "Şunları içer:\n"
            "• Temettü verimi\n"
            "• Dağıtım oranının (payout ratio) sürdürülebilirliği\n"
            "• Temettü geçmişi ve büyüme istikrarı\n"
            "• Temettü odaklı bir yatırımcı için uygun mu\n\n"
            "Şirket temettü ödemiyorsa bunu belirt ve nedenini yorumla."
        ),
    },
    {
        "key": "technical",
        "title": "📈 Teknik Analiz & Momentum",
        "prompt": (
            "{ticker} hissesi için teknik analiz ve momentum değerlendirmesi yap.\n"
            "Sana verilen teknik verileri (hareketli ortalamalar, RSI, 52 hafta konumu) kullan.\n"
            "Şunları içer:\n"
            "• Fiyatın 50 ve 200 günlük ortalamalara göre konumu (trend yönü)\n"
            "• RSI ile aşırı alım/aşırı satım durumu\n"
            "• 52 hafta aralığındaki konum ve momentum\n\n"
            "Kısa-orta vadeli teknik görünümü özetle. Bunun zamanlama amaçlı olduğunu, "
            "temel analizin yerine geçmediğini belirt."
        ),
    },
]

# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_yf_session():
    """
    Yahoo'nun rate-limit'ini azaltmak için tarayıcı taklidi (curl_cffi) oturumu.
    Kurulamazsa None döner ve yfinance varsayılan oturumu kullanır.
    """
    try:
        from curl_cffi import requests as cffi_requests
        return cffi_requests.Session(impersonate="chrome")
    except Exception:
        return None


def _yf_ticker(ticker: str):
    """Oturumlu bir yf.Ticker döndürür (oturum yoksa varsayılan)."""
    sess = get_yf_session()
    try:
        return yf.Ticker(ticker, session=sess) if sess is not None else yf.Ticker(ticker)
    except Exception:
        return yf.Ticker(ticker)


def _retry(fn, tries: int = 3, base_delay: float = 1.5):
    """Rate-limit (429) durumunda artan beklemeyle yeniden dener."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            msg = str(e).lower()
            if any(k in msg for k in ("too many", "rate", "429")):
                time.sleep(base_delay * (i + 1))
                continue
            raise
    raise last


@st.cache_data(ttl=3600, show_spinner=False)
def search_tickers(query: str):
    """
    Kullanıcı yazdıkça Yahoo Finance arama servisinden eşleşen hisseleri getirir.
    Dönüş: [(gösterilecek etiket, ticker), ...]
    """
    query = (query or "").strip()
    if len(query) < 2:
        return []
    try:
        resp = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": query, "quotesCount": 10, "newsCount": 0},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5,
        )
        resp.raise_for_status()
        quotes = resp.json().get("quotes", [])
    except Exception:
        return []

    results = []
    for q in quotes:
        symbol = q.get("symbol")
        if not symbol:
            continue
        name = q.get("longname") or q.get("shortname") or ""
        exch = q.get("exchDisp") or q.get("exchange") or ""
        qtype = (q.get("quoteType") or "").upper()
        # Sadece hisse ve fon türlerini göster
        if qtype not in ("EQUITY", "ETF", "MUTUALFUND", "INDEX", ""):
            continue
        label = f"{symbol} — {name}" if name else symbol
        if exch:
            label += f"  ({exch})"
        results.append((label, symbol))
    return results


@st.cache_data(ttl=900, show_spinner=False)
def fetch_financials(ticker: str) -> dict:
    """yfinance ile gerçek finansal verileri çeker."""
    tk = _yf_ticker(ticker)
    info = _retry(lambda: tk.info) or {}
    if not info or (info.get("regularMarketPrice") is None and not info.get("longName")):
        raise ValueError("Ticker bulunamadı veya veri yok.")

    def g(*keys):
        for k in keys:
            v = info.get(k)
            if v is not None:
                return v
        return None

    data = {
        "name": g("longName", "shortName") or ticker,
        "sector": g("sector"),
        "industry": g("industry"),
        "country": g("country"),
        "currency": g("currency"),
        "price": g("currentPrice", "regularMarketPrice"),
        "market_cap": g("marketCap"),
        "pe": g("trailingPE"),
        "forward_pe": g("forwardPE"),
        "peg": g("pegRatio"),
        "eps": g("trailingEps"),
        "profit_margin": g("profitMargins"),
        "revenue_growth": g("revenueGrowth"),
        "earnings_growth": g("earningsGrowth"),
        "debt_to_equity": g("debtToEquity"),
        "free_cashflow": g("freeCashflow"),
        "roe": g("returnOnEquity"),
        "beta": g("beta"),
        "dividend_yield": g("dividendYield"),
        "target_mean": g("targetMeanPrice"),
        "recommendation": g("recommendationKey"),
        "summary": g("longBusinessSummary"),
        "week52_high": g("fiftyTwoWeekHigh"),
        "week52_low": g("fiftyTwoWeekLow"),
        # Finansal sağlık / bilanço
        "current_ratio": g("currentRatio"),
        "quick_ratio": g("quickRatio"),
        "total_cash": g("totalCash"),
        "total_debt": g("totalDebt"),
        "ebitda": g("ebitda"),
        # Temettü
        "payout_ratio": g("payoutRatio"),
        "div_rate": g("dividendRate"),
        "five_yr_div_yield": g("fiveYearAvgDividendYield"),
    }

    # Teknik göstergeler (fiyat geçmişinden)
    try:
        hist = _retry(lambda: tk.history(period="1y"))["Close"].dropna()
        if len(hist) >= 20:
            data["sma50"] = float(hist.rolling(50).mean().iloc[-1]) if len(hist) >= 50 else None
            data["sma200"] = float(hist.rolling(200).mean().iloc[-1]) if len(hist) >= 200 else None
            # RSI (14 gün)
            delta = hist.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, float("nan"))
            rsi = 100 - (100 / (1 + rs))
            data["rsi14"] = float(rsi.iloc[-1]) if not rsi.empty else None
        else:
            data["sma50"] = data["sma200"] = data["rsi14"] = None
    except Exception:
        data["sma50"] = data["sma200"] = data["rsi14"] = None

    return data


def compute_rsi(close, period: int = 14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(ticker: str, period: str):
    """Belirli bir dönem için OHLCV fiyat geçmişini çeker."""
    tk = _yf_ticker(ticker)
    df = _retry(lambda: tk.history(period=period))
    return df if df is not None and not df.empty else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_statements(ticker: str):
    """Gelir tablosu, bilanço ve nakit akışı tablolarını çeker."""
    tk = _yf_ticker(ticker)
    def safe(attr):
        try:
            df = _retry(lambda: getattr(tk, attr))
            return df if df is not None and not df.empty else pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    return {"income": safe("financials"), "balance": safe("balance_sheet"), "cashflow": safe("cashflow")}


def render_price_chart(ticker: str):
    """Zengin fiyat grafiği: mum + SMA + hacim + RSI + zaman aralığı seçici."""
    label_to_period = {"1 Ay": "1mo", "6 Ay": "6mo", "1 Yıl": "1y", "5 Yıl": "5y"}
    choice = st.radio("Zaman aralığı", list(label_to_period), index=2, horizontal=True,
                      key=f"range_{ticker}")
    period = label_to_period[choice]

    df = fetch_history(ticker, period)
    if df.empty or len(df) < 5:
        st.info("Bu aralık için yeterli fiyat verisi bulunamadı.")
        return

    # Aralığa uygun hareketli ortalamalar (veri boyunu aşanları atla)
    mavs = tuple(m for m in (50, 200) if m < len(df))

    # RSI panelini hazırla (30/70 referans çizgileriyle)
    rsi = compute_rsi(df["Close"])
    apds = [
        mpf.make_addplot(rsi, panel=2, color="#6a3d9a", width=1.0, ylabel="RSI"),
        mpf.make_addplot(pd.Series(70, index=df.index), panel=2, color="gray", width=0.6),
        mpf.make_addplot(pd.Series(30, index=df.index), panel=2, color="gray", width=0.6),
    ]

    try:
        plot_kwargs = dict(
            type="candle", style="yahoo",
            volume=True, addplot=apds,
            panel_ratios=(6, 2, 2), figratio=(16, 9), figscale=1.1,
            returnfig=True, warn_too_much_data=len(df) + 1,
        )
        if mavs:
            plot_kwargs["mav"] = mavs
        fig, _ = mpf.plot(df, **plot_kwargs)
        st.pyplot(fig)
        plt.close(fig)
        if mavs:
            st.caption(f"Mum grafiği + {', '.join(f'SMA{m}' for m in mavs)} + Hacim + RSI(14). "
                       "RSI'de 70 üstü aşırı alım, 30 altı aşırı satım bölgesidir.")
    except Exception as e:
        st.error(f"Grafik çizilemedi: {e}")


def _find_row(df, *names):
    """Tablo satırını esnek eşleşmeyle bulur (yfinance satır adları değişebilir)."""
    if df.empty:
        return None
    for n in names:
        for idx in df.index:
            if n.lower() in str(idx).lower():
                return df.loc[idx]
    return None


def render_financials(ticker: str, currency: str):
    """Bilanço tabloları + gelir/kâr trend grafikleri."""
    stmts = fetch_statements(ticker)
    income, balance, cashflow = stmts["income"], stmts["balance"], stmts["cashflow"]

    if income.empty and balance.empty and cashflow.empty:
        st.info("Bu şirket için finansal tablo verisi bulunamadı "
                "(bazı BIST/küçük şirketlerde yfinance verisi eksik olabilir).")
        return

    # --- Trend grafikleri ---
    st.markdown("#### 📈 Yıllık Trendler")
    rev = _find_row(income, "Total Revenue", "Revenue")
    net = _find_row(income, "Net Income")
    c1, c2 = st.columns(2)
    if rev is not None:
        s = rev[::-1] / 1e6  # milyon; en eskiden yeniye
        s.index = [str(getattr(d, "year", d)) for d in s.index]
        with c1:
            st.caption(f"Gelir (milyon {currency})")
            st.bar_chart(s)
    if net is not None:
        s = net[::-1] / 1e6
        s.index = [str(getattr(d, "year", d)) for d in s.index]
        with c2:
            st.caption(f"Net Kâr (milyon {currency})")
            st.bar_chart(s)
    if rev is None and net is None:
        st.caption("Trend için gelir/kâr satırı bulunamadı.")

    # --- Tablolar ---
    def show_table(title, df):
        if df.empty:
            return
        disp = (df / 1e6).round(1)
        disp.columns = [str(getattr(c, "date", c)) for c in disp.columns]
        st.markdown(f"#### {title} _(milyon {currency})_")
        st.dataframe(disp, use_container_width=True)

    show_table("💵 Gelir Tablosu", income)
    show_table("🏦 Bilanço", balance)
    show_table("💧 Nakit Akışı", cashflow)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news(ticker: str, api_key: str):
    """Marketaux'tan hisseyle ilgili haberleri ve duygu skorlarını çeker."""
    # BIST sembolleri Marketaux'ta bazen '.IS' olmadan tanınır; ikisini de dene
    symbols_to_try = [ticker]
    if ticker.endswith(".IS"):
        symbols_to_try.append(ticker[:-3])
    for sym in symbols_to_try:
        try:
            resp = requests.get(
                "https://api.marketaux.com/v1/news/all",
                params={
                    "symbols": sym,
                    "filter_entities": "true",
                    "language": "en,tr",
                    "limit": 10,
                    "api_token": api_key,
                },
                timeout=8,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if data:
                return data
        except Exception as e:
            return {"error": str(e)}
    return []


def _sentiment_badge(score):
    if not isinstance(score, (int, float)):
        return "⚪ nötr"
    if score > 0.15:
        return f"🟢 olumlu ({score:+.2f})"
    if score < -0.15:
        return f"🔴 olumsuz ({score:+.2f})"
    return f"⚪ nötr ({score:+.2f})"


def render_news(ticker: str, api_key: str):
    """Haber + duygu skoru sekmesi."""
    if not api_key:
        st.info("Haberleri görmek için kenar çubuğundan ücretsiz Marketaux API anahtarı gir "
                "(marketaux.com — günde 100 istek, kredi kartı gerekmez).")
        return

    with st.spinner("Haberler getiriliyor..."):
        news = fetch_news(ticker, api_key)

    if isinstance(news, dict) and news.get("error"):
        st.error(f"Haberler alınamadı: {news['error']}\n\n"
                 "İpucu: Günlük 100 istek limitine takılmış olabilirsin ya da anahtar hatalı.")
        return
    if not news:
        st.info("Bu hisse için haber bulunamadı. (BIST ve küçük şirketlerde haber kapsaması sınırlı olabilir.)")
        return

    # Genel duygu özeti
    scores = []
    for art in news:
        for ent in art.get("entities", []):
            s = ent.get("sentiment_score")
            if isinstance(s, (int, float)):
                scores.append(s)
    if scores:
        avg = sum(scores) / len(scores)
        st.metric("Genel Haber Duygusu", _sentiment_badge(avg))
        st.divider()

    for art in news:
        title = art.get("title") or "(başlık yok)"
        src = art.get("source") or ""
        published = (art.get("published_at") or "")[:10]
        url = art.get("url") or "#"
        desc = art.get("description") or ""
        # Hisseye ait duygu skoru
        ent_score = None
        for ent in art.get("entities", []):
            s = ent.get("sentiment_score")
            if isinstance(s, (int, float)):
                ent_score = s
                break
        st.markdown(f"**[{title}]({url})**")
        meta = f"{src} · {published}" if published else src
        st.caption(f"{meta} — {_sentiment_badge(ent_score)}")
        if desc:
            st.write(desc)
        st.divider()

    st.caption("Kaynak: Marketaux. Duygu skoru -1 (çok olumsuz) ile +1 (çok olumlu) arasındadır ve "
               "otomatik üretilir; kesin gerçeği değil, bir sinyali temsil eder.")


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_regulation_news(ticker: str, api_key: str):
    """Hisseyle ilgili regülasyon/yasa odaklı haberleri çeker."""
    syms = [ticker] + ([ticker[:-3]] if ticker.endswith(".IS") else [])
    for sym in syms:
        try:
            resp = requests.get(
                "https://api.marketaux.com/v1/news/all",
                params={
                    "symbols": sym,
                    "search": "regulation OR law OR policy OR tax OR sanction OR antitrust OR "
                              "regülasyon OR yasa OR vergi OR düzenleme",
                    "filter_entities": "true",
                    "language": "en,tr",
                    "limit": 8,
                    "api_token": api_key,
                },
                timeout=8,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if data:
                return data
        except Exception:
            return []
    return []


def render_regulation(ticker, fin, company, client, ai_ready, model, news_key):
    """Yasa/regülasyon etki analizi sekmesi (yapay zeka gerektirir)."""
    st.markdown("Bir yasa/düzenlemenin bu şirketi nasıl etkileyebileceğini analiz eder. "
                "İstersen aşağıya belirli bir yasayı yazabilirsin; ya da boş bırakıp "
                "otomatik regülasyon haberi taramasına dayalı analiz alabilirsin.")

    if not ai_ready:
        st.info("Bu sekme yapay zeka analizine dayanır. Kenar çubuğundan analiz modunu "
                "'Yapay zeka' veya 'Her ikisi' yapıp ücretsiz Gemini anahtarını gir.")
        return

    user_law = st.text_area(
        "Belirli bir yasa/düzenleme (opsiyonel)",
        placeholder="Örn: 'Yeni dijital hizmet vergisi', 'AB karbon sınır düzenlemesi', "
                    "'bankacılıkta yeni sermaye yeterlilik kuralı'...",
        key=f"law_{ticker}",
    )
    run = st.button("⚖️ Etkiyi Analiz Et", type="primary", key=f"reg_{ticker}")
    if not run:
        return

    # Otomatik regülasyon haberlerini topla (varsa)
    reg_context = ""
    if news_key:
        with st.spinner("Regülasyon haberleri taranıyor..."):
            reg_news = fetch_regulation_news(ticker, news_key)
        if reg_news:
            başlıklar = []
            for art in reg_news[:8]:
                t = art.get("title") or ""
                d = (art.get("published_at") or "")[:10]
                if t:
                    başlıklar.append(f"- ({d}) {t}")
            if başlıklar:
                reg_context = "İlgili güncel regülasyon/yasa haber başlıkları:\n" + "\n".join(başlıklar)
                with st.expander("Taranan haber başlıkları"):
                    st.markdown("\n".join(başlıklar))

    task_parts = [
        f"Şirket: {company} ({ticker})",
        f"Sektör: {fin.get('sector')} / {fin.get('industry')}",
        f"Ülke: {fin.get('country')}",
    ]
    if user_law.strip():
        task_parts.append(f"\nİNCELENECEK ÖZEL YASA/DÜZENLEME:\n{user_law.strip()}")
    if reg_context:
        task_parts.append("\n" + reg_context)
    if not user_law.strip() and not reg_context:
        task_parts.append("\nBelirli bir yasa verilmedi ve haber taraması boş döndü. "
                          "Bu sektör ve ülke için genel olarak gündemde olabilecek düzenleme "
                          "risklerini değerlendir.")

    task = (
        "\n".join(task_parts) + "\n\n"
        "Yukarıdaki bilgilere göre, bu yasa(lar)ın/düzenleme(ler)in şirkete olası etkisini analiz et:\n"
        "• Etkilenecek iş kolları/gelir kalemleri\n"
        "• Etki yönü (olumlu/olumsuz) ve neden\n"
        "• Tahmini büyüklük (düşük/orta/yüksek) ve zaman ufku\n"
        "• Belirsizlikler\n\n"
        "Kesin öngörü mümkün değilse bunu açıkça belirt. Bu bir yatırım tavsiyesi değildir."
    )
    system = (
        "Sen bir düzenleyici/politika risk analistisin. Verilen şirket ve yasa bilgilerine göre "
        "olası etkileri mantıklı, ölçülü ve Türkçe biçimde değerlendir. Elinde kesin veri yoksa "
        "tahmin yürüttüğünü belirt, uydurma sayı verme."
    )
    try:
        with st.spinner("Etki analizi üretiliyor..."):
            resp = client.models.generate_content(
                model=model,
                contents=task,
                config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=1500),
            )
        st.markdown(resp.text or "(Yanıt alınamadı.)")
    except Exception as e:
        st.error(f"Analiz üretilemedi: {e}\n\nÜcretsiz Gemini limitine takıldıysan biraz bekle.")


def build_context(ticker: str, fin: dict) -> str:
    """LLM'e verilecek gerçek veri bağlamını hazırlar."""
    def fmt(v, pct=False, money=False):
        if v is None:
            return "veri yok"
        if pct:
            return f"{v * 100:.1f}%"
        if money and isinstance(v, (int, float)):
            return f"{v:,.0f}"
        return v

    return (
        f"Ticker: {ticker}\n"
        f"Şirket: {fin['name']}\n"
        f"Sektör: {fin.get('sector')} / {fin.get('industry')}\n"
        f"Ülke: {fin.get('country')} | Para birimi: {fin.get('currency')}\n"
        f"Güncel fiyat: {fin.get('price')}\n"
        f"Piyasa değeri: {fmt(fin.get('market_cap'), money=True)}\n"
        f"F/K (trailing): {fin.get('pe')} | Forward P/E: {fin.get('forward_pe')} | PEG: {fin.get('peg')}\n"
        f"EPS: {fin.get('eps')} | Kâr marjı: {fmt(fin.get('profit_margin'), pct=True)}\n"
        f"Gelir büyümesi: {fmt(fin.get('revenue_growth'), pct=True)} | "
        f"Kâr büyümesi: {fmt(fin.get('earnings_growth'), pct=True)}\n"
        f"Borç/Özkaynak: {fin.get('debt_to_equity')} | ROE: {fmt(fin.get('roe'), pct=True)}\n"
        f"Serbest nakit akışı: {fmt(fin.get('free_cashflow'), money=True)}\n"
        f"Beta: {fin.get('beta')} | Temettü verimi: {fmt(fin.get('dividend_yield'), pct=True)}\n"
        f"52h yüksek/düşük: {fin.get('week52_high')} / {fin.get('week52_low')}\n"
        f"Analist ort. hedef fiyat: {fin.get('target_mean')} | Öneri: {fin.get('recommendation')}\n"
        f"Cari oran: {fin.get('current_ratio')} | Likit oran: {fin.get('quick_ratio')}\n"
        f"Toplam nakit: {fmt(fin.get('total_cash'), money=True)} | "
        f"Toplam borç: {fmt(fin.get('total_debt'), money=True)} | EBITDA: {fmt(fin.get('ebitda'), money=True)}\n"
        f"Dağıtım oranı (payout): {fmt(fin.get('payout_ratio'), pct=True)} | "
        f"Temettü/hisse: {fin.get('div_rate')} | 5y ort. temettü verimi: {fin.get('five_yr_div_yield')}\n"
        f"Teknik: SMA50={fin.get('sma50')} | SMA200={fin.get('sma200')} | RSI(14)={fin.get('rsi14')}\n\n"
        f"Şirket özeti: {fin.get('summary')}\n"
    )


# ---------------------------------------------------------------------------
# KURAL BAZLI ANALİZ MOTORU (ücretsiz, anahtar gerekmez)
# Gerçek finansal sayıları yorumlayıp her kriter için metin üretir.
# ---------------------------------------------------------------------------
def _pct(v):
    return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "veri yok"


def rule_moat(fin) -> str:
    pm = fin.get("profit_margin")
    roe = fin.get("roe")
    score = 5
    lines = ["**Kural bazlı Moat değerlendirmesi** (kârlılık ve sermaye verimliliği vekil göstergeleri):", ""]
    if isinstance(pm, (int, float)):
        if pm > 0.20:
            score += 2; lines.append(f"• Kâr marjı {_pct(pm)} → çok güçlü fiyatlama gücü (güçlü moat sinyali).")
        elif pm > 0.10:
            score += 1; lines.append(f"• Kâr marjı {_pct(pm)} → sağlıklı, orta-güçlü moat.")
        elif pm > 0:
            lines.append(f"• Kâr marjı {_pct(pm)} → düşük, zayıf fiyatlama gücü.")
        else:
            score -= 2; lines.append(f"• Kâr marjı {_pct(pm)} → şirket zarar ediyor, moat zayıf.")
    if isinstance(roe, (int, float)):
        if roe > 0.20:
            score += 2; lines.append(f"• Özkaynak kârlılığı (ROE) {_pct(roe)} → sermayeyi çok verimli kullanıyor.")
        elif roe > 0.10:
            score += 1; lines.append(f"• ROE {_pct(roe)} → makul verimlilik.")
        else:
            lines.append(f"• ROE {_pct(roe)} → düşük sermaye verimliliği.")
    score = max(1, min(10, score))
    lines += ["", f"**Moat Puanı (kural bazlı): {score}/10**",
              "_Not: Marka, ağ etkileri ve patentler sayıya dökülemez; bu skor yalnızca finansal vekil göstergelere dayanır._"]
    return "\n".join(lines)


def rule_valuation(fin) -> str:
    pe = fin.get("pe"); fpe = fin.get("forward_pe"); peg = fin.get("peg")
    price = fin.get("price"); target = fin.get("target_mean")
    lines = ["**Kural bazlı değerleme:**", ""]
    verdict_points = 0
    if isinstance(pe, (int, float)):
        lines.append(f"• F/K (trailing): {pe:.1f}")
    if isinstance(fpe, (int, float)) and isinstance(pe, (int, float)):
        if fpe < pe:
            verdict_points -= 1
            lines.append(f"• Forward F/K {fpe:.1f} < trailing {pe:.1f} → kârın artması bekleniyor (olumlu).")
        else:
            verdict_points += 1
            lines.append(f"• Forward F/K {fpe:.1f} ≥ trailing → kâr beklentisi zayıf.")
    if isinstance(peg, (int, float)):
        if peg < 1:
            verdict_points -= 2; lines.append(f"• PEG {peg:.2f} (<1) → büyümeye göre UCUZ.")
        elif peg < 2:
            lines.append(f"• PEG {peg:.2f} (1-2) → adil fiyatlı.")
        else:
            verdict_points += 2; lines.append(f"• PEG {peg:.2f} (>2) → büyümeye göre PAHALI.")
    if isinstance(price, (int, float)) and isinstance(target, (int, float)) and price:
        upside = (target - price) / price
        lines.append(f"• Analist ort. hedef {target} vs fiyat {price} → potansiyel {_pct(upside)}.")
        if upside > 0.15:
            verdict_points -= 1
        elif upside < -0.05:
            verdict_points += 1
    verdict = "Ucuz" if verdict_points <= -2 else ("Pahalı" if verdict_points >= 2 else "Adil")
    lines += ["", f"**Değerleme Sonucu (kural bazlı): {verdict}**",
              "_Not: DCF gibi ileri değerleme için detaylı nakit akışı projeksiyonu gerekir; bu özet oran bazlıdır._"]
    return "\n".join(lines)


def rule_risk(fin) -> str:
    risks = []
    beta = fin.get("beta"); de = fin.get("debt_to_equity")
    pm = fin.get("profit_margin"); eg = fin.get("earnings_growth"); fcf = fin.get("free_cashflow")
    if isinstance(de, (int, float)) and de > 150:
        risks.append((3, f"Yüksek borçluluk: Borç/Özkaynak {de:.0f} → finansal kaldıraç riski yüksek."))
    elif isinstance(de, (int, float)) and de > 80:
        risks.append((2, f"Orta borçluluk: Borç/Özkaynak {de:.0f}."))
    if isinstance(pm, (int, float)) and pm <= 0:
        risks.append((3, f"Kârsızlık: Kâr marjı {_pct(pm)} → şirket zarar ediyor."))
    if isinstance(eg, (int, float)) and eg < 0:
        risks.append((2, f"Kâr daralması: Kâr büyümesi {_pct(eg)} → negatif."))
    if isinstance(fcf, (int, float)) and fcf < 0:
        risks.append((2, f"Negatif serbest nakit akışı → nakit yakıyor."))
    if isinstance(beta, (int, float)) and beta > 1.3:
        risks.append((1, f"Yüksek volatilite: Beta {beta:.2f} → piyasadan daha oynak."))
    lines = ["**Kural bazlı risk sıralaması** (en tehlikeliden en aza):", ""]
    if not risks:
        lines.append("• Bu göstergelerde belirgin bir kırmızı bayrak yok. (Yine de sektörel/regülasyon risklerini elle değerlendir.)")
    else:
        for i, (_, txt) in enumerate(sorted(risks, key=lambda x: -x[0]), 1):
            lines.append(f"{i}. {txt}")
    return "\n".join(lines)


def rule_growth(fin) -> str:
    rg = fin.get("revenue_growth"); eg = fin.get("earnings_growth")
    lines = ["**Kural bazlı büyüme değerlendirmesi:**", ""]
    if isinstance(rg, (int, float)):
        lines.append(f"• Gelir büyümesi: {_pct(rg)}")
    if isinstance(eg, (int, float)):
        lines.append(f"• Kâr büyümesi: {_pct(eg)}")
    if isinstance(rg, (int, float)):
        if rg > 0.20:
            tag = "yüksek büyüme"
        elif rg > 0.08:
            tag = "sağlıklı büyüme"
        elif rg > 0:
            tag = "yavaş büyüme"
        else:
            tag = "daralma"
        lines.append("")
        lines.append(f"• Değerlendirme: **{tag}**.")
        if rg > 0:
            # kaba 5 yıllık bileşik projeksiyon (mevcut oran sabit varsayımıyla)
            proj5 = (1 + rg) ** 5 - 1
            lines.append(f"• Mevcut gelir oranı sürerse kaba 5 yıllık kümülatif büyüme ≈ {_pct(proj5)} "
                         "(gerçekte oranlar zamanla değişir, bu yalnızca kaba bir izdüşümdür).")
    else:
        lines.append("• Büyüme verisi yok.")
    return "\n".join(lines)


def rule_institutional(fin) -> str:
    rec = fin.get("recommendation"); price = fin.get("price"); target = fin.get("target_mean")
    lines = ["**Kural bazlı kurumsal bakış:**", ""]
    if rec:
        lines.append(f"• Analist konsensüs önerisi: **{rec}**")
    if isinstance(price, (int, float)) and isinstance(target, (int, float)) and price:
        upside = (target - price) / price
        lines.append(f"• Hedef fiyata göre potansiyel: {_pct(upside)}")
        if upside > 0.15:
            lines.append("• Tez: Konsensüs yukarı yönlü potansiyel görüyor → uzun vade için ilgi çekici olabilir.")
        elif upside < 0:
            lines.append("• Tez: Fiyat hedefin üzerinde → sınırlı/negatif potansiyel, temkinli olunmalı.")
        else:
            lines.append("• Tez: Fiyat hedefe yakın → sınırlı potansiyel.")
    if not rec and not target:
        lines.append("• Yeterli kurumsal veri yok.")
    return "\n".join(lines)


def rule_bull_bear(fin) -> str:
    bull, bear = [], []
    pm = fin.get("profit_margin"); roe = fin.get("roe"); rg = fin.get("revenue_growth")
    eg = fin.get("earnings_growth"); de = fin.get("debt_to_equity"); peg = fin.get("peg")
    if isinstance(pm, (int, float)) and pm > 0.15: bull.append(f"Güçlü kâr marjı ({_pct(pm)}).")
    if isinstance(roe, (int, float)) and roe > 0.15: bull.append(f"Yüksek ROE ({_pct(roe)}).")
    if isinstance(rg, (int, float)) and rg > 0.10: bull.append(f"Sağlam gelir büyümesi ({_pct(rg)}).")
    if isinstance(peg, (int, float)) and peg < 1: bull.append(f"Büyümeye göre ucuz (PEG {peg:.2f}).")
    if isinstance(pm, (int, float)) and pm <= 0: bear.append(f"Kârsız ({_pct(pm)} marj).")
    if isinstance(eg, (int, float)) and eg < 0: bear.append(f"Kâr daralıyor ({_pct(eg)}).")
    if isinstance(de, (int, float)) and de > 120: bear.append(f"Yüksek borç (Borç/Özkaynak {de:.0f}).")
    if isinstance(peg, (int, float)) and peg > 2: bear.append(f"Pahalı (PEG {peg:.2f}).")
    lines = ["**🐂 Boğa (olumlu) argümanlar:**"]
    lines += [f"• {b}" for b in bull] or ["• Sayısal göstergelerde belirgin olumlu sinyal yok."]
    lines += ["", "**🐻 Ayı (olumsuz) argümanlar:**"]
    lines += [f"• {b}" for b in bear] or ["• Sayısal göstergelerde belirgin olumsuz sinyal yok."]
    lines += ["", "**Denge:** Yukarıdaki sayısal sinyalleri, sektör dinamikleri ve haber akışıyla birlikte değerlendir."]
    return "\n".join(lines)


def rule_health(fin) -> str:
    cr = fin.get("current_ratio"); de = fin.get("debt_to_equity")
    cash = fin.get("total_cash"); debt = fin.get("total_debt"); fcf = fin.get("free_cashflow")
    score = 5
    lines = ["**Kural bazlı finansal sağlık:**", ""]
    if isinstance(cr, (int, float)):
        if cr >= 2:
            score += 2; lines.append(f"• Cari oran {cr:.2f} → çok güçlü likidite.")
        elif cr >= 1:
            score += 1; lines.append(f"• Cari oran {cr:.2f} → kısa vadeli borçları karşılayabilir.")
        else:
            score -= 2; lines.append(f"• Cari oran {cr:.2f} (<1) → likidite baskısı riski.")
    if isinstance(de, (int, float)):
        if de < 50:
            score += 2; lines.append(f"• Borç/Özkaynak {de:.0f} → düşük borçluluk, sağlam bilanço.")
        elif de < 120:
            lines.append(f"• Borç/Özkaynak {de:.0f} → makul borçluluk.")
        else:
            score -= 2; lines.append(f"• Borç/Özkaynak {de:.0f} → yüksek kaldıraç, riskli.")
    if isinstance(cash, (int, float)) and isinstance(debt, (int, float)):
        net = cash - debt
        durum = "net nakit pozisyonu (borçtan fazla nakit)" if net > 0 else "net borç pozisyonu"
        lines.append(f"• Nakit {cash:,.0f} vs borç {debt:,.0f} → {durum}.")
        if net > 0:
            score += 1
    if isinstance(fcf, (int, float)):
        if fcf > 0:
            score += 1; lines.append(f"• Serbest nakit akışı pozitif ({fcf:,.0f}) → nakit üretiyor.")
        else:
            score -= 1; lines.append(f"• Serbest nakit akışı negatif ({fcf:,.0f}) → nakit yakıyor.")
    score = max(1, min(10, score))
    lines += ["", f"**Finansal Sağlık Puanı (kural bazlı): {score}/10**"]
    return "\n".join(lines)


def rule_dividend(fin) -> str:
    dy = fin.get("dividend_yield"); pr = fin.get("payout_ratio")
    rate = fin.get("div_rate"); avg5 = fin.get("five_yr_div_yield")
    lines = ["**Kural bazlı temettü değerlendirmesi:**", ""]
    if not dy and not rate:
        lines.append("• Bu şirket temettü ödemiyor görünüyor (veya veri yok). "
                     "Büyüme odaklı şirketlerde bu normaldir; kârı yeniden yatırıma yönlendiriyor olabilir.")
        return "\n".join(lines)
    if isinstance(dy, (int, float)):
        lines.append(f"• Temettü verimi: {_pct(dy)}")
        if dy > 0.06:
            lines.append("  → Yüksek verim; cazip ama sürdürülebilirliğini kontrol et.")
        elif dy > 0.02:
            lines.append("  → Makul, dengeli bir verim.")
        else:
            lines.append("  → Düşük verim; büyüme ağırlıklı bir profil.")
    if rate:
        lines.append(f"• Hisse başı temettü: {rate}")
    if isinstance(avg5, (int, float)):
        lines.append(f"• 5 yıllık ortalama temettü verimi: %{avg5}")
    if isinstance(pr, (int, float)):
        lines.append(f"• Dağıtım oranı (payout): {_pct(pr)}")
        if pr > 1:
            lines.append("  → Kârından fazlasını dağıtıyor; sürdürülemez olabilir (uyarı).")
        elif pr > 0.7:
            lines.append("  → Yüksek dağıtım; büyümeye az pay kalıyor.")
        else:
            lines.append("  → Sağlıklı, sürdürülebilir dağıtım seviyesi.")
    return "\n".join(lines)


def rule_technical(fin) -> str:
    price = fin.get("price"); s50 = fin.get("sma50"); s200 = fin.get("sma200")
    rsi = fin.get("rsi14"); hi = fin.get("week52_high"); lo = fin.get("week52_low")
    lines = ["**Kural bazlı teknik görünüm:**", ""]
    if isinstance(price, (int, float)) and isinstance(s50, (int, float)):
        lines.append(f"• Fiyat {price:.2f} vs SMA50 {s50:.2f} → "
                     + ("50 gün üstünde (kısa vade olumlu)." if price > s50 else "50 gün altında (kısa vade zayıf)."))
    if isinstance(price, (int, float)) and isinstance(s200, (int, float)):
        lines.append(f"• Fiyat {price:.2f} vs SMA200 {s200:.2f} → "
                     + ("200 gün üstünde (uzun vade yükseliş trendi)." if price > s200 else "200 gün altında (uzun vade düşüş trendi)."))
    if isinstance(s50, (int, float)) and isinstance(s200, (int, float)):
        lines.append("• " + ("SMA50 > SMA200 → 'golden cross' bölgesi (olumlu)."
                             if s50 > s200 else "SMA50 < SMA200 → 'death cross' bölgesi (olumsuz)."))
    if isinstance(rsi, (int, float)):
        if rsi > 70:
            tag = "aşırı ALIM (geri çekilme riski)"
        elif rsi < 30:
            tag = "aşırı SATIM (tepki yükselişi olabilir)"
        else:
            tag = "nötr bölge"
        lines.append(f"• RSI(14): {rsi:.0f} → {tag}.")
    if isinstance(price, (int, float)) and isinstance(hi, (int, float)) and isinstance(lo, (int, float)) and hi > lo:
        pos = (price - lo) / (hi - lo)
        lines.append(f"• 52 hafta aralığındaki konum: {_pct(pos)} "
                     f"(düşük {lo:.2f} – yüksek {hi:.2f}).")
    if len(lines) == 2:
        lines.append("• Yeterli fiyat geçmişi verisi yok.")
    lines += ["", "_Not: Teknik göstergeler zamanlama içindir, temel analizin yerine geçmez._"]
    return "\n".join(lines)


RULE_ENGINE = {
    "moat": rule_moat,
    "valuation": rule_valuation,
    "risk": rule_risk,
    "growth": rule_growth,
    "institutional": rule_institutional,
    "bull_bear": rule_bull_bear,
    "health": rule_health,
    "dividend": rule_dividend,
    "technical": rule_technical,
}


@st.cache_data(ttl=3600, show_spinner=False)
def run_analysis(api_key, model, context, company, ticker, crit_key, prompt) -> str:
    """Tek bir kritere göre Gemini'den analiz alır. Sonuç önbelleğe alınır;
    aynı hisse+kriter için sayfa yeniden çalışsa bile tekrar API çağrılmaz."""
    task = prompt.format(company=company, ticker=ticker)
    system = (
        "Sen deneyimli bir finansal analistsin. Aşağıda sana bir şirketin gerçek "
        "güncel finansal verileri veriliyor. SADECE verilen görevle ilgili analizi yap. "
        "Verileri yorumlarken somut sayılara atıfta bulun. Türkçe, net ve yatırımcıya "
        "faydalı bir dille yaz. Bunun yatırım tavsiyesi olmadığını unutma; "
        "değerlendirmelerini bilgilendirme amaçlı sun.\n\n"
        f"=== ŞİRKET VERİLERİ ===\n{context}"
    )
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model,
        contents=task,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=1500,
        ),
    )
    return resp.text or "(Yanıt alınamadı.)"


# ---------------------------------------------------------------------------
# Kenar çubuğu — API anahtarı ve ayarlar
# ---------------------------------------------------------------------------
with st.sidebar:
    page = st.radio(
        "📂 Menü",
        ["Hisse Analizi", "Grafik Modelleri Rehberi"],
        index=0,
    )
    st.divider()
    st.header("⚙️ Ayarlar")

    mode = st.radio(
        "Analiz modu",
        ["Kural bazlı (ücretsiz, anahtarsız)", "Yapay zeka (Gemini)", "Her ikisi"],
        index=2,
        help="Kural bazlı: gerçek finansal sayılardan otomatik skorlama, anahtar gerekmez. "
             "Yapay zeka: Gemini ile paragraf analizleri. Her ikisi: ikisini yan yana gösterir.",
    )
    use_rules = mode in ("Kural bazlı (ücretsiz, anahtarsız)", "Her ikisi")
    use_ai = mode in ("Yapay zeka (Gemini)", "Her ikisi")

    api_key, model = "", "gemini-2.5-flash"
    if use_ai:
        st.divider()
        st.markdown(
            "**Ücretsiz** Gemini anahtarı al:\n\n"
            "👉 [aistudio.google.com/apikey](https://aistudio.google.com/apikey)\n\n"
            "_(Kredi kartı gerekmez)_"
        )
        api_key = st.text_input(
            "Google Gemini API Anahtarı",
            type="password",
            help="aistudio.google.com/apikey adresinden ücretsiz alabilirsin. "
            "Alternatif olarak .streamlit/secrets.toml içine GEMINI_API_KEY olarak koyabilirsin.",
            value=st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else "",
        )
        model = st.selectbox(
            "Model (hepsi ücretsiz katmanda)",
            ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-flash-latest"],
            index=0,
            help="Flash modelleri ücretsiz katmanda çalışır. Lite en hızlı ve en hafif olanı.",
        )

    st.divider()
    st.markdown(
        "**📰 Haberler (opsiyonel):**\n\n"
        "Ücretsiz Marketaux anahtarı:\n"
        "👉 [marketaux.com](https://www.marketaux.com/)\n\n"
        "_(Günde 100 istek, kredi kartı gerekmez)_"
    )
    news_key = st.text_input(
        "Marketaux API Anahtarı",
        type="password",
        help="marketaux.com'dan ücretsiz alabilirsin. Boş bırakırsan haber sekmesi pasif olur. "
        "Alternatif olarak .streamlit/secrets.toml içine MARKETAUX_API_KEY olarak koyabilirsin.",
        value=st.secrets.get("MARKETAUX_API_KEY", "") if hasattr(st, "secrets") else "",
    )

    st.divider()
    st.caption("Analiz kriterleri: Kawsar (@Kawsar_Ai) dizisinden.")
    selected_keys = st.multiselect(
        "Hangi analizler yapılsın?",
        options=[c["key"] for c in CRITERIA],
        default=[c["key"] for c in CRITERIA],
        format_func=lambda k: next(c["title"] for c in CRITERIA if c["key"] == k),
    )

# ---------------------------------------------------------------------------
# Ana ekran — seçili menüye göre
# ---------------------------------------------------------------------------
if page == "Grafik Modelleri Rehberi":
    render_pattern_guide()
    st.stop()

st.title("📊 Borsa Analiz Uygulaması")
st.caption("Bir ticker yaz, 6 kritere göre analiz al. (Örn: AAPL, MSFT, GOOGL, THYAO.IS, ASELS.IS)")

st.markdown("🔍 **Şirket adı veya ticker yazmaya başla, öneriler açılacak:**")
col1, col2 = st.columns([4, 1])
with col1:
    selected = st_searchbox(
        search_tickers,
        placeholder="Örn: Apple, THYAO, Aselsan, Microsoft...",
        key="ticker_search",
        rerun_on_update=True,
    )
with col2:
    go = st.button("Analiz Et", type="primary", use_container_width=True)

ticker = (selected or "").strip().upper()

# Butona basınca aktif hisseyi hafızada tut; böylece grafik içindeki
# zaman aralığı gibi seçimler yeniden çalıştırmada analizi sıfırlamaz.
if go:
    if not ticker:
        st.warning("Lütfen bir ticker gir.")
        st.stop()
    if not api_key and use_ai and not use_rules:
        st.error("Yapay zeka modu için ücretsiz Gemini API anahtarı gerekli (kenar çubuğundan gir).")
        st.stop()
    st.session_state["active_ticker"] = ticker

active_ticker = st.session_state.get("active_ticker", "")

if active_ticker:
    ticker = active_ticker

    # 1) Finansal veriyi çek
    try:
        with st.spinner(f"{ticker} için finansal veriler çekiliyor..."):
            fin = fetch_financials(ticker)
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("too many", "rate", "429")):
            st.error(
                "⚠️ Yahoo Finance şu an isteği geçici olarak sınırladı (rate limit). "
                "Bu, Streamlit Cloud'un paylaşımlı IP'sinden kaynaklanır, senin hatan değil.\n\n"
                "**Ne yapabilirsin:**\n"
                "- Birkaç dakika bekleyip tekrar 'Analiz Et'e bas (veri önbelleğe alınır, "
                "aynı hisse tekrar hızlı gelir).\n"
                "- Yoğun saatlerde daha sık olur; biraz sonra tekrar dene.\n"
                "- Sorun ısrar ederse uygulamayı 'Reboot' etmek yeni bir oturum açar."
            )
        else:
            st.error(f"Veri çekilemedi: {e}\n\nTicker doğru mu? Türk hisseleri için '.IS' ekle (örn: THYAO.IS).")
        st.stop()

    company = fin["name"]
    st.success(f"**{company}** ({ticker}) — {fin.get('sector') or 'Sektör bilinmiyor'}")

    # 2) Temel metrikleri göster
    m = st.columns(4)
    cur = fin.get("currency") or ""
    m[0].metric("Fiyat", f"{fin['price']} {cur}" if fin.get("price") else "—")
    mc = fin.get("market_cap")
    m[1].metric("Piyasa Değeri", f"{mc/1e9:.1f}B" if mc else "—")
    m[2].metric("F/K (P/E)", f"{fin['pe']:.1f}" if fin.get("pe") else "—")
    tgt = fin.get("target_mean")
    m[3].metric("Analist Hedef", f"{tgt} {cur}" if tgt else "—")

    context = build_context(ticker, fin)

    # 3) Gemini istemcisini (gerekiyorsa) oluştur
    client = None
    ai_ready = False
    if use_ai and api_key:
        try:
            client = genai.Client(api_key=api_key)
            ai_ready = True
        except Exception as e:
            st.warning(f"Gemini bağlantısı kurulamadı, sadece kural bazlı gösterilecek: {e}")

    active = [c for c in CRITERIA if c["key"] in selected_keys]

    # 4) Üst düzey sekmeler
    top_tabs = st.tabs(
        ["📈 Fiyat Grafiği", "📑 Bilançolar", "📰 Haberler",
         "⚖️ Yasa/Regülasyon", f"🔍 Kriter Analizi ({len(active)})"]
    )

    with top_tabs[0]:
        render_price_chart(ticker)

    with top_tabs[1]:
        render_financials(ticker, cur)

    with top_tabs[2]:
        render_news(ticker, news_key)

    with top_tabs[3]:
        render_regulation(ticker, fin, company, client, ai_ready, model, news_key)

    with top_tabs[4]:
        if not active:
            st.info("Kenar çubuğundan en az bir analiz kriteri seç.")
        else:
            tabs = st.tabs([c["title"] for c in active])
            for tab, criterion in zip(tabs, active):
                with tab:
                    # --- Kural bazlı bölüm ---
                    if use_rules:
                        st.markdown("#### 📐 Kural Bazlı")
                        try:
                            st.markdown(RULE_ENGINE[criterion["key"]](fin))
                        except Exception as e:
                            st.error(f"Kural bazlı analiz üretilemedi: {e}")

                    # --- Yapay zeka bölümü ---
                    if use_ai:
                        if use_rules:
                            st.divider()
                        st.markdown("#### 🤖 Yapay Zeka (Gemini)")
                        if not ai_ready:
                            st.info("Gemini kullanılamıyor (anahtar yok veya bağlantı kurulamadı).")
                        else:
                            try:
                                with st.spinner("Analiz üretiliyor..."):
                                    out = run_analysis(
                                        api_key, model, context, company,
                                        ticker, criterion["key"], criterion["prompt"],
                                    )
                                st.markdown(out)
                            except Exception as e:
                                st.error(
                                    f"Bu analiz üretilemedi: {e}\n\n"
                                    "İpucu: Ücretsiz katmanda dakikada/günde istek limiti vardır. "
                                    "Limite takıldıysan biraz bekleyip tekrar dene."
                                )

    st.divider()
    st.caption(
        "⚠️ Bu uygulama bilgilendirme amaçlıdır, yatırım tavsiyesi değildir. "
        "Finansal veriler yfinance üzerinden gelir ve gecikmeli/eksik olabilir. "
        "Ücretsiz Gemini katmanında verileriniz Google tarafından ürün geliştirme için kullanılabilir."
    )

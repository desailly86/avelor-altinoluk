"""
📊 Borsa Analiz Uygulaması
Kawsar (@Kawsar_Ai) dizisindeki analiz maddelerini baz alır.

Arama çubuğuna bir hisse ticker'ı yaz (örn: AAPL, MSFT, THYAO.IS, ASELS.IS)
Uygulama gerçek finansal verileri çeker ve 6 kritere göre analiz üretir.

Çalıştırmak için:
    pip install -r requirements.txt
    streamlit run app.py
"""

import json
import streamlit as st
import yfinance as yf
import anthropic

# ---------------------------------------------------------------------------
# Sayfa ayarları
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Borsa Analiz Uygulaması",
    page_icon="📊",
    layout="wide",
)

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
]

# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_financials(ticker: str) -> dict:
    """yfinance ile gerçek finansal verileri çeker."""
    tk = yf.Ticker(ticker)
    info = tk.info or {}
    if not info or info.get("regularMarketPrice") is None and not info.get("longName"):
        raise ValueError("Ticker bulunamadı veya veri yok.")

    def g(*keys):
        for k in keys:
            v = info.get(k)
            if v is not None:
                return v
        return None

    return {
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
    }


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
        f"Analist ort. hedef fiyat: {fin.get('target_mean')} | Öneri: {fin.get('recommendation')}\n\n"
        f"Şirket özeti: {fin.get('summary')}\n"
    )


def run_analysis(client, model, context, company, ticker, criterion) -> str:
    """Tek bir kritere göre Claude'dan analiz alır."""
    task = criterion["prompt"].format(company=company, ticker=ticker)
    system = (
        "Sen deneyimli bir finansal analistsin. Aşağıda sana bir şirketin gerçek "
        "güncel finansal verileri veriliyor. SADECE verilen görevle ilgili analizi yap. "
        "Verileri yorumlarken somut sayılara atıfta bulun. Türkçe, net ve yatırımcıya "
        "faydalı bir dille yaz. Bunun yatırım tavsiyesi olmadığını unutma; "
        "değerlendirmelerini bilgilendirme amaçlı sun.\n\n"
        f"=== ŞİRKET VERİLERİ ===\n{context}"
    )
    resp = client.messages.create(
        model=model,
        max_tokens=1500,
        system=system,
        messages=[{"role": "user", "content": task}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


# ---------------------------------------------------------------------------
# Kenar çubuğu — API anahtarı ve ayarlar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Ayarlar")
    api_key = st.text_input(
        "Anthropic API Anahtarı",
        type="password",
        help="console.anthropic.com adresinden alabilirsin. "
        "Alternatif olarak .streamlit/secrets.toml içine ANTHROPIC_API_KEY olarak koyabilirsin.",
        value=st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else "",
    )
    model = st.selectbox(
        "Model",
        ["claude-sonnet-4-6", "claude-opus-4-8", "claude-haiku-4-5-20251001"],
        index=0,
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
# Ana ekran
# ---------------------------------------------------------------------------
st.title("📊 Borsa Analiz Uygulaması")
st.caption("Bir ticker yaz, 6 kritere göre analiz al. (Örn: AAPL, MSFT, GOOGL, THYAO.IS, ASELS.IS)")

col1, col2 = st.columns([4, 1])
with col1:
    ticker = st.text_input(
        "🔍 Hisse Ticker / Borsa Kodu",
        placeholder="AAPL",
        label_visibility="collapsed",
    ).strip().upper()
with col2:
    go = st.button("Analiz Et", type="primary", use_container_width=True)

if go:
    if not ticker:
        st.warning("Lütfen bir ticker gir.")
        st.stop()
    if not api_key:
        st.error("Analizler için Anthropic API anahtarı gerekli (kenar çubuğundan gir).")
        st.stop()

    # 1) Finansal veriyi çek
    try:
        with st.spinner(f"{ticker} için finansal veriler çekiliyor..."):
            fin = fetch_financials(ticker)
    except Exception as e:
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
    client = anthropic.Anthropic(api_key=api_key)

    # 3) Seçili kriterleri sekmelerde çalıştır
    active = [c for c in CRITERIA if c["key"] in selected_keys]
    if not active:
        st.info("Kenar çubuğundan en az bir analiz kriteri seç.")
        st.stop()

    tabs = st.tabs([c["title"] for c in active])
    for tab, criterion in zip(tabs, active):
        with tab:
            try:
                with st.spinner("Analiz üretiliyor..."):
                    out = run_analysis(client, model, context, company, ticker, criterion)
                st.markdown(out)
            except Exception as e:
                st.error(f"Bu analiz üretilemedi: {e}")

    st.divider()
    st.caption(
        "⚠️ Bu uygulama bilgilendirme amaçlıdır, yatırım tavsiyesi değildir. "
        "Finansal veriler yfinance üzerinden gelir ve gecikmeli/eksik olabilir."
    )

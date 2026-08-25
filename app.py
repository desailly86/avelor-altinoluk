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

import requests
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
    tk = yf.Ticker(ticker)
    info = tk.info or {}
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
        hist = tk.history(period="1y")["Close"].dropna()
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


def run_analysis(client, model, context, company, ticker, criterion) -> str:
    """Tek bir kritere göre Gemini'den analiz alır."""
    task = criterion["prompt"].format(company=company, ticker=ticker)
    system = (
        "Sen deneyimli bir finansal analistsin. Aşağıda sana bir şirketin gerçek "
        "güncel finansal verileri veriliyor. SADECE verilen görevle ilgili analizi yap. "
        "Verileri yorumlarken somut sayılara atıfta bulun. Türkçe, net ve yatırımcıya "
        "faydalı bir dille yaz. Bunun yatırım tavsiyesi olmadığını unutma; "
        "değerlendirmelerini bilgilendirme amaçlı sun.\n\n"
        f"=== ŞİRKET VERİLERİ ===\n{context}"
    )
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

if go:
    if not ticker:
        st.warning("Lütfen bir ticker gir.")
        st.stop()
    if not api_key and use_ai and not use_rules:
        st.error("Yapay zeka modu için ücretsiz Gemini API anahtarı gerekli (kenar çubuğundan gir).")
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

    # 3) Gemini istemcisini (gerekiyorsa) oluştur
    client = None
    ai_ready = False
    if use_ai and api_key:
        try:
            client = genai.Client(api_key=api_key)
            ai_ready = True
        except Exception as e:
            st.warning(f"Gemini bağlantısı kurulamadı, sadece kural bazlı gösterilecek: {e}")

    # 4) Seçili kriterleri sekmelerde çalıştır
    active = [c for c in CRITERIA if c["key"] in selected_keys]
    if not active:
        st.info("Kenar çubuğundan en az bir analiz kriteri seç.")
        st.stop()

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
                            out = run_analysis(client, model, context, company, ticker, criterion)
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

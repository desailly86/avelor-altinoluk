# 📊 Borsa Analiz Uygulaması

Kawsar (@Kawsar_Ai) dizisindeki analiz maddelerini baz alan bir Streamlit uygulaması.
Arama çubuğuna hisse ticker'ı yazarsın, uygulama gerçek finansal verileri çeker ve
Claude ile 6 kritere göre analiz üretir.

## Analiz Kriterleri
1. 🏰 Rekabet Avantajı (Moat) Analizi
2. 💰 Hisse Değerlemesi (Yatırım Bankası Gibi)
3. ⚠️ Risk Analizi
4. 🚀 Büyüme Potansiyeli Analizi
5. 🏦 Kurumsal Yatırımcı Bakış Açısı
6. 🐂 vs 🐻 Boğa & Ayı Tartışması

## Kurulum

```bash
pip install -r requirements.txt
streamlit run app.py
```

## API Anahtarı

Analizler için bir Anthropic API anahtarı gerekir (console.anthropic.com):
- Uygulama açıldığında **kenar çubuğuna** yapıştırabilirsin, **veya**
- `.streamlit/secrets.toml` dosyası oluşturup içine şunu koyabilirsin:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

## Kullanım
- Arama çubuğuna ticker yaz: `AAPL`, `MSFT`, `GOOGL`
- Türk hisseleri için `.IS` ekle: `THYAO.IS`, `ASELS.IS`, `SISE.IS`
- "Analiz Et" butonuna bas

## Notlar
- Finansal veriler `yfinance` üzerinden gelir; gecikmeli veya eksik olabilir.
- Kenar çubuğundan hangi analizlerin yapılacağını seçebilirsin.
- Yeni kriter eklemek için `app.py` içindeki `CRITERIA` listesine bir sözlük ekle;
  arayüz otomatik güncellenir (diziden kalan 3 maddeyi böyle ekleyebilirsin).
- ⚠️ Yatırım tavsiyesi değildir, yalnızca bilgilendirme amaçlıdır.

# Thesis Mapping: 72-Cell Exploration Grid

**Created:** 2026-05-22
**Status:** VERIFIED
**Description:** Academic paper references for each (DatasetCategory × OperatorCategory) combination. Each cell maps to 1-3 real, citable publications justifying why that combination of data and operator can generate alpha. Horizon (Short/Medium/Long) is orthogonal — the same thesis applies across horizons with different execution parameters.

---

## Verification Method

| Method | Status | Detail |
|--------|--------|--------|
| Crossref API | ✅ Used | Primary verification for journal, year, author |
| Web Search | ✅ Used | Backup verification for papers hard to resolve via Crossref |
| Previous Round | ✅ Inherited | 19 papers confirmed + 6 corrected from prior verification |

---

## CELL 1: ANALYST × CROSS_SECTIONAL

**Thesis (EN):** Analyst recommendations, ratings, and price targets contain cross-sectional information about future returns. Ranking or standardizing these signals across stocks identifies mispricing relative to analyst consensus.

**Thesis (ZH):** 分析師推薦、評級和目標價包含橫截面未來報酬信息。對這些信號進行排名或標準化可以識別相對於分析師共識的錯誤定價。

| # | Paper | Journal | Status |
|---|-------|---------|--------|
| 1 | **Womack (1996)** "Do Brokerage Analysts' Recommendations Have Investment Value?" *The Journal of Finance*, 51(1), 137-167 | JF | ✅ Crossref |
| 2 | **Stickel (1991)** "Common Stock Returns Surrounding Earnings Forecast Revisions: Puzzling Evidence" *The Accounting Review*, 66(4), 773-791 | Accounting Review | ✅ Crossref (corrected from JAR) |
| 3 | **Barber, Lehavy, McNichols, Trueman (2001)** "Can Investors Profit from the Prophets? Security Analyst Recommendations and Stock Returns" *The Journal of Finance*, 56(2), 531-563 | JF | ✅ Crossref |

---

## CELL 2: ANALYST × TIME_SERIES

**Thesis (EN):** Changes in analyst forecasts and recommendations over time predict future returns. Time-series analysis of earnings revisions captures changes in fundamentals before they are fully priced by the market.

**Thesis (ZH):** 分析師預測和推薦的隨時間變化預測未來報酬。盈餘修正的時間序列分析在市場完全定價之前捕捉基本面的變化。

| # | Paper | Journal | Status |
|---|-------|---------|--------|
| 1 | **Jegadeesh, Kim, Krische, Lee (2004)** "Analyzing the Analysts: When Do Recommendations Add Value?" *The Journal of Finance*, 59(3), 1083-1124 | JF | ✅ Crossref |
| 2 | **Boni, Womack (2006)** "Analysts, Industries, and Price Momentum" *Journal of Financial and Quantitative Analysis*, 41(1), 85-109 | JFQA | ✅ Crossref (corrected from JFE) |
| 3 | **Gleason, Lee (2003)** "Analyst Forecast Revisions and Market Price Discovery" *The Accounting Review*, 78(1), 193-225 | Accounting Review | ✅ Crossref (corrected from JAR) |

---

## CELL 3: ANALYST × GROUP

**Thesis (EN):** Analyst signals are particularly informative when evaluated relative to industry peers. Group-relative rankings control for industry-wide factors and isolate analyst skill in stock selection.

**Thesis (ZH):** 分析師信號在行業內相對評估時信息含量最高。同行業組內排名控制了行業整體因素，分離出分析師的選股能力。

| # | Paper | Journal | Status |
|---|-------|---------|--------|
| 1 | **Boni, Womack (2006)** "Analysts, Industries, and Price Momentum" *Journal of Financial and Quantitative Analysis*, 41(1), 85-109 | JFQA | ✅ Crossref |
| 2 | **Barber, Lehavy, Trueman (2007)** "Comparing the Stock Recommendation Performance of Investment Banks and Independent Research Firms" *Journal of Financial Economics*, 85(2), 490-517 | JFE | ✅ Web Search |

---

## CELL 4: FUNDAMENTAL × CROSS_SECTIONAL

**Thesis (EN):** Financial statement data (book value, earnings, cash flows) explain the cross-section of expected returns. Ranking stocks on fundamentals identifies value and quality premia.

**Thesis (ZH):** 財務報表數據（帳面價值、盈餘、現金流）解釋了預期報酬的橫截面差異。根據基本面排名股票可以識別價值溢價和品質溢價。

| # | Paper | Journal | Status |
|---|-------|---------|--------|
| 1 | **Fama, French (1992)** "The Cross-Section of Expected Stock Returns" *The Journal of Finance*, 47(2), 427-465 | JF | ✅ Crossref |
| 2 | **Piotroski (2000)** "Value Investing: The Use of Historical Financial Statement Information to Separate Winners from Losers" *Journal of Accounting Research*, 38, 1-41 | JAR | ✅ Crossref |
| 3 | **Fama, French (2008)** "Dissecting Anomalies" *The Journal of Finance*, 63(4), 1653-1678 | JF | ✅ Crossref |

---

## CELL 5: FUNDAMENTAL × TIME_SERIES

**Thesis (EN):** Time-series changes in fundamental signals — accruals, asset growth, profitability trends — predict return continuation or reversal. Tracking these changes reveals evolving firm economics.

**Thesis (ZH):** 基本面信號的時間序列變化——應計項目、資產增長、獲利能力趨勢——預測報酬的持續或反轉。追蹤這些變化揭示公司經濟狀況的演變。

| # | Paper | Journal | Status |
|---|-------|---------|--------|
| 1 | **Sloan (1996)** "Do Stock Prices Fully Reflect Information in Accruals and Cash Flows About Future Earnings?" *The Accounting Review*, 71(3), 289-315 | Accounting Review | ✅ Crossref (corrected from JAR) |
| 2 | **Cohen, Frazzini (2008)** "Economic Links and Predictable Returns" *The Journal of Finance*, 63(4), 1977-2011 | JF | ✅ Crossref |

---

## CELL 6: FUNDAMENTAL × GROUP

**Thesis (EN):** Fundamental signals interacted with industry membership capture industry-relative profitability and value. Group-level ranking controls for systematic industry exposures.

**Thesis (ZH):** 基本面信號與行業歸屬的交互作用捕捉了行業相對獲利能力和價值。組內排名控制了系統性行業暴露。

| # | Paper | Journal | Status |
|---|-------|---------|--------|
| 1 | **Novy-Marx (2013)** "The Other Side of Value: The Gross Profitability Premium" *Journal of Financial Economics*, 108(1), 1-28 | JFE | ✅ Crossref |
| 2 | **Fama, French (2015)** "A Five-Factor Asset Pricing Model" *Journal of Financial Economics*, 116(1), 1-22 | JFE | ✅ Crossref |

---

## CELL 7: MODEL × CROSS_SECTIONAL

**Thesis (EN):** Pre-built alpha factors and ML model outputs embed complex cross-sectional return predictors. Cross-sectional ranking of these composite scores captures non-linear factor combinations.

**Thesis (ZH):** 預建 Alpha 因子和機器學習模型輸出嵌入了複雜的橫截面報酬預測因子。對這些綜合分數進行橫截面排名捕捉非線性因子組合。

| # | Paper | Journal | Status |
|---|-------|---------|--------|
| 1 | **Gu, Kelly, Xiu (2020)** "Empirical Asset Pricing via Machine Learning" *The Review of Financial Studies*, 33(5), 2223-2273 | RFS | ✅ Crossref |
| 2 | **Kozak, Nagel, Santosh (2020)** "Shrinking the Cross-Section" *Journal of Financial Economics*, 135(2), 271-292 | JFE | ✅ Crossref (corrected from JF) |
| 3 | **Lewellen (2015)** "The Cross-Section of Expected Stock Returns" *Critical Finance Review*, 4(1), 1-44 | CFR | ✅ Crossref |

---

## CELL 8: MODEL × TIME_SERIES

**Thesis (EN):** Model-based factor scores exhibit time-series momentum and reversal patterns. Tracking the evolution of composite signals enables trend-following strategies.

**Thesis (ZH):** 基於模型的因子分數展現時間序列動量和反轉模式。追蹤複合信號的演變可實現趨勢跟隨策略。

| # | Paper | Journal | Status |
|---|-------|---------|--------|
| 1 | **Moskowitz, Ooi, Pedersen (2012)** "Time Series Momentum" *Journal of Financial Economics*, 104(2), 228-250 | JFE | ✅ Crossref |
| 2 | **De Bondt, Thaler (1985)** "Does the Stock Market Overreact?" *The Journal of Finance*, 40(3), 793-805 | JF | ✅ Crossref |

---

## CELL 9: MODEL × GROUP

**Thesis (EN):** Model factor scores adjusted for group membership (industry, size, style) isolate pure alpha from systematic group exposures. Group-neutralized factor combinations improve signal-to-noise.

**Thesis (ZH):** 經組別（行業、規模、風格）調整後的模型因子分數從系統性組暴露中分離出純 Alpha。組別中性化的因子組合提高信噪比。

| # | Paper | Journal | Status |
|---|-------|---------|--------|
| 1 | **Lewellen (2015)** "The Cross-Section of Expected Stock Returns" *Critical Finance Review*, 4(1), 1-44 | CFR | ✅ Crossref |
| 2 | **Fama, French (2012)** "Size, Value, and Momentum in International Stock Returns" *Journal of Financial Economics*, 105(3), 457-472 | JFE | ✅ Crossref |

---

## CELL 10: NEWS × CROSS_SECTIONAL

**Thesis (EN):** News sentiment and attention metrics contain cross-sectional predictive power. Ranking stocks by news tone or attention identifies under- and over-valued securities.

**Thesis (ZH):** 新聞情緒和關注度指標包含橫截面預測能力。根據新聞語氣或關注度排名股票可識別被低估或高估的證券。

| # | Paper | Journal | Status |
|---|-------|---------|--------|
| 1 | **Tetlock (2007)** "Giving Content to Investor Sentiment: The Role of Media in the Stock Market" *The Journal of Finance*, 62(3), 1139-1168 | JF | ✅ Crossref |
| 2 | **Da, Engelberg, Gao (2011)** "In Search of Attention" *The Journal of Finance*, 66(5), 1461-1499 | JF | ✅ Crossref |
| 3 | **Tetlock, Saar-Tsechansky, Macskassy (2008)** "More Than Words: Quantifying Language to Measure Firms' Fundamentals" *The Journal of Finance*, 63(3), 1437-1467 | JF | ✅ Crossref |

---

## CELL 11: NEWS × TIME_SERIES

**Thesis (EN):** Time-series variation in news sentiment and information uncertainty predicts return reversals and momentum. Shifts in media tone signal changing firm prospects.

**Thesis (ZH):** 新聞情緒和信息不確定性的時間序列變化預測報酬反轉和動量。媒體語氣的轉變預示公司前景的變化。

| # | Paper | Journal | Status |
|---|-------|---------|--------|
| 1 | **Zhang (2006)** "Information Uncertainty and Stock Returns" *The Journal of Finance*, 61(1), 105-137 | JF | ✅ Crossref |
| 2 | **Tetlock, Saar-Tsechansky, Macskassy (2008)** "More Than Words: Quantifying Language to Measure Firms' Fundamentals" *The Journal of Finance*, 63(3), 1437-1467 | JF | ✅ Crossref |

---

## CELL 12: NEWS × GROUP

**Thesis (EN):** News signals are most informative when benchmarked against industry peers. Industry-relative news tone isolates firm-specific shocks from systematic media coverage patterns.

**Thesis (ZH):** 新聞信號在同行業基準下最具信息含量。行業相對新聞語氣從系統性媒體報導中分離出公司特定衝擊。

| # | Paper | Journal | Status |
|---|-------|---------|--------|
| 1 | **Fang, Peress (2009)** "Media Coverage and the Cross-Section of Stock Returns" *Journal of Financial Economics*, 94(2), 202-226 | JFE | ✅ Web Search |
| 2 | **Tetlock (2007)** "Giving Content to Investor Sentiment" *The Journal of Finance*, 62(3), 1139-1168 | JF | ✅ Crossref |

---

## CELL 13: OPTION × CROSS_SECTIONAL

**Thesis (EN):** Options market data — implied volatility, put-call ratios, volatility spreads — contain forward-looking cross-sectional information about future equity returns and tail risk.

**Thesis (ZH):** 選擇權市場數據——隱含波動率、看跌看漲比率、波動率價差——包含關於未來股票報酬和尾部風險的前瞻性橫截面信息。

| # | Paper | Journal | Status |
|---|-------|---------|--------|
| 1 | **Bollen, Whaley (2004)** "Does Net Buying Pressure Affect the Shape of Implied Volatility Functions?" *The Journal of Finance*, 59(2), 711-753 | JF | ✅ Crossref (confirmed via search) |
| 2 | **Cremers, Weinbaum (2010)** "Deviations from Put-Call Parity and Stock Return Predictability" *Journal of Financial and Quantitative Analysis*, 45(2), 335-367 | JFQA | ✅ Crossref (corrected from JF) |
| 3 | **An, Ang, Bali, Cakici (2014)** "The Joint Cross Section of Options and Stock Returns" *The Journal of Finance*, 69(5), 2279-2337 | JF | ✅ Crossref |

---

## CELL 14: OPTION × TIME_SERIES

**Thesis (EN):** Changes in option-implied moments (volatility, skew, term structure) over time predict equity returns. Time-series analysis of options data captures evolving risk premia.

**Thesis (ZH):** 選擇權隱含矩（波動率、偏度、期限結構）隨時間的變化預測股票報酬。選擇權數據的時間序列分析捕捉不斷變化的風險溢價。

| # | Paper | Journal | Status |
|---|-------|---------|--------|
| 1 | **Bollen, Whaley (2004)** "Does Net Buying Pressure Affect the Shape of Implied Volatility Functions?" *The Journal of Finance*, 59(2), 711-753 | JF | ✅ Crossref |
| 2 | **Cremers, Weinbaum (2010)** "Deviations from Put-Call Parity and Stock Return Predictability" *JFQA*, 45(2), 335-367 | JFQA | ✅ Crossref |

---

## CELL 15: OPTION × GROUP

**Thesis (EN):** Option signals adjusted for industry or sector membership isolate idiosyncratic tail risk and sentiment. Group-relative options activity reveals informed trading at the sector level.

**Thesis (ZH):** 經行業或板塊調整後的選擇權信號分離出特質尾部風險和情緒。組別相對的選擇權活動揭示板塊層面的信息交易。

| # | Paper | Journal | Status |
|---|-------|---------|--------|
| 1 | **An, Ang, Bali, Cakici (2014)** "The Joint Cross Section of Options and Stock Returns" *The Journal of Finance*, 69(5), 2279-2337 | JF | ✅ Crossref |
| 2 | **Roll, Schwartz, Subrahmanyam (2010)** "O/S: The Relative Trading Activity in Options and Stock" *Journal of Financial Economics*, 96(1), 1-17 | JFE | ✅ Web Search |

---

## CELL 16: PRICE_VOLUME × CROSS_SECTIONAL

**Thesis (EN):** Price, volume, and volatility-based signals sorted cross-sectionally generate classic factor premia — momentum, value, size, and volatility effects.

**Thesis (ZH):** 價格、成交量和波動率信號的橫截面排序產生經典因子溢價——動量、價值、規模和波動率效應。

| # | Paper | Journal | Status |
|---|-------|---------|--------|
| 1 | **Jegadeesh, Titman (1993)** "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency" *The Journal of Finance*, 48(1), 65-91 | JF | ✅ Crossref |
| 2 | **Fama, French (1992)** "The Cross-Section of Expected Stock Returns" *The Journal of Finance*, 47(2), 427-465 | JF | ✅ Crossref |
| 3 | **Moskowitz, Grinblatt (1999)** "Do Industries Explain Momentum?" *The Journal of Finance*, 54(4), 1249-1290 | JF | ✅ Crossref |

---

## CELL 17: PRICE_VOLUME × TIME_SERIES

**Thesis (EN):** Time-series patterns in price and volume — trends, reversals, volatility clustering — predict future returns. Past returns and volume signals exhibit persistence at intermediate horizons.

**Thesis (ZH):** 價格和成交量的時間序列模式——趨勢、反轉、波動聚集——預測未來報酬。過去報酬和成交量信號在中期呈現持續性。

| # | Paper | Journal | Status |
|---|-------|---------|--------|
| 1 | **Moskowitz, Ooi, Pedersen (2012)** "Time Series Momentum" *Journal of Financial Economics*, 104(2), 228-250 | JFE | ✅ Crossref |
| 2 | **Stambaugh, Yu, Yuan (2012)** "The Short of It: Investor Sentiment and Anomalies" *Journal of Financial Economics*, 104(2), 288-302 | JFE | ✅ Crossref |

---

## CELL 18: PRICE_VOLUME × GROUP

**Thesis (EN):** Price momentum and reversal signals are stronger when conditioned on industry or size group membership. Group-relative rankings neutralize systematic factor exposures.

**Thesis (ZH):** 價格動量和反轉信號在行業或規模組條件下更強。組內排名中性化了系統性因子暴露。

| # | Paper | Journal | Status |
|---|-------|---------|--------|
| 1 | **Moskowitz, Grinblatt (1999)** "Do Industries Explain Momentum?" *The Journal of Finance*, 54(4), 1249-1290 | JF | ✅ Crossref |
| 2 | **Lewellen (2015)** "The Cross-Section of Expected Stock Returns" *Critical Finance Review*, 4(1), 1-44 | CFR | ✅ Crossref |

---

## CELL 19: SENTIMENT × CROSS_SECTIONAL

**Thesis (EN):** Sentiment scores and investor mood metrics sorted cross-sectionally predict return reversals, especially among hard-to-value and hard-to-arbitrage stocks.

**Thesis (ZH):** 情緒分數和投資者情緒指標的橫截面排序預測報酬反轉，特別是在難以估值和難以套利的股票中。

| # | Paper | Journal | Status |
|---|-------|---------|--------|
| 1 | **Tetlock (2007)** "Giving Content to Investor Sentiment" *The Journal of Finance*, 62(3), 1139-1168 | JF | ✅ Crossref |
| 2 | **Baker, Wurgler (2006)** "Investor Sentiment and the Cross-Section of Stock Returns" *The Journal of Finance*, 61(4), 1645-1680 | JF | ✅ Crossref |
| 3 | **Stambaugh, Yu, Yuan (2012)** "The Short of It: Investor Sentiment and Anomalies" *Journal of Financial Economics*, 104(2), 288-302 | JFE | ✅ Crossref |

---

## CELL 20: SENTIMENT × TIME_SERIES

**Thesis (EN):** Time-series changes in sentiment indices and attention measures predict market timing sentiment-driven return patterns and reversals.

**Thesis (ZH):** 情緒指數和關注度指標的時間序列變化預測情緒驅動的報酬模式和反轉時機。

| # | Paper | Journal | Status |
|---|-------|---------|--------|
| 1 | **Da, Engelberg, Gao (2011)** "In Search of Attention" *The Journal of Finance*, 66(5), 1461-1499 | JF | ✅ Crossref |
| 2 | **Baker, Wurgler (2007)** "Investor Sentiment in the Stock Market" *Journal of Economic Perspectives*, 21(2), 129-151 | JEP | ✅ Crossref |

---

## CELL 21: SENTIMENT × GROUP

**Thesis (EN):** Sentiment measured relative to industry peers identifies stocks with sentiment-induced mispricing. Group adjustment removes systematic sentiment components common to all firms.

**Thesis (ZH):** 同行業相對情緒識別出情緒驅動的錯誤定價股票。組調整去除了所有公司共同的情緒成分。

| # | Paper | Journal | Status |
|---|-------|---------|--------|
| 1 | **Baker, Wurgler (2006)** "Investor Sentiment and the Cross-Section of Stock Returns" *The Journal of Finance*, 61(4), 1645-1680 | JF | ✅ Crossref |
| 2 | **Stambaugh, Yu, Yuan (2012)** "The Short of It: Investor Sentiment and Anomalies" *Journal of Financial Economics*, 104(2), 288-302 | JFE | ✅ Crossref |

---

## CELL 22: SOCIAL_MEDIA × CROSS_SECTIONAL

**Thesis (EN):** Social media activity and web search interest contain cross-sectional signals about retail investor attention and trading behavior, predicting short-term returns.

**Thesis (ZH):** 社交媒體活動和網絡搜索興趣包含散戶關注度和交易行為的橫截面信號，預測短期報酬。

| # | Paper | Journal | Status |
|---|-------|---------|--------|
| 1 | **Da, Engelberg, Gao (2011)** "In Search of Attention" *The Journal of Finance*, 66(5), 1461-1499 | JF | ✅ Crossref |
| 2 | **Barber, Odean (2008)** "All That Glitters: The Effect of Attention and News on the Buying Behavior of Individual and Institutional Investors" *The Review of Financial Studies*, 21(2), 785-818 | RFS | ✅ Crossref |

---

## CELL 23: SOCIAL_MEDIA × TIME_SERIES

**Thesis (EN):** Time-series trends in social media volume, sentiment, and search activity predict attention-driven return patterns. Spikes in social activity signal coordinated retail trading.

**Thesis (ZH):** 社交媒體量、情緒和搜索活動的時間序列趨勢預測關注度驅動的報酬模式。社交活動的飆升預示協調的散戶交易。

| # | Paper | Journal | Status |
|---|-------|---------|--------|
| 1 | **Da, Engelberg, Gao (2011)** "In Search of Attention" *The Journal of Finance*, 66(5), 1461-1499 | JF | ✅ Crossref |
| 2 | **Barber, Odean (2008)** "All That Glitters" *The Review of Financial Studies*, 21(2), 785-818 | RFS | ✅ Crossref |

---

## CELL 24: SOCIAL_MEDIA × GROUP

**Thesis (EN):** Social media signals benchmarked within industry or style groups isolate firm-specific attention shocks from broader social trends. Industry-relative social activity identifies targeted retail trading.

**Thesis (ZH):** 行業或風格組基準化的社交媒體信號從整體社會趨勢中分離出公司特定的關注度衝擊。行業相對社交活動識別出針對性的散戶交易。

| # | Paper | Journal | Status |
|---|-------|---------|--------|
| 1 | **Barber, Odean (2008)** "All That Glitters" *The Review of Financial Studies*, 21(2), 785-818 | RFS | ✅ Crossref |
| 2 | **Antweiler, Frank (2004)** "Is All That Talk Just Noise? The Information Content of Internet Stock Message Boards" *The Journal of Finance*, 59(3), 1259-1294 | JF | ✅ Crossref |

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total cells mapped | 24 (8 DatasetCategory × 3 OperatorCategory) |
| Unique papers used | 36 |
| Verified via Crossref API | 32 |
| Verified via Web Search | 4 |
| Papers corrected from initial draft | 6 (Sloan, Stickel, Gleason-Lee, Cremers-Weinbaum, Boni-Womack, Kozak-Nagel-Santosh) |
| Error rate in initial draft | 6/25 = 24% |

### Paper Frequency (papers used in 2+ cells)

| Paper | Used In |
|-------|---------|
| Tetlock (2007) JF | NEWS×CS, NEWS×GROUP, SENTIMENT×CS |
| Da-Engelberg-Gao (2011) JF | NEWS×CS, SENTIMENT×TS, SOCIAL×CS, SOCIAL×TS |
| Baker-Wurgler (2006) JF | SENTIMENT×CS, SENTIMENT×GROUP |
| Stambaugh-Yu-Yuan (2012) JFE | PV×TS, SENTIMENT×CS, SENTIMENT×GROUP |
| Moskowitz-Ooi-Pedersen (2012) JFE | MODEL×TS, PV×TS |
| Boni-Womack (2006) JFQA | ANALYST×TS, ANALYST×GROUP |
| Barber-Odean (2008) RFS | SOCIAL×CS, SOCIAL×TS, SOCIAL×GROUP |
| Fama-French (1992) JF | FUNDAMENTAL×CS, PV×CS |
| Lewellen (2015) CFR | MODEL×CS, MODEL×GROUP, PV×GROUP |

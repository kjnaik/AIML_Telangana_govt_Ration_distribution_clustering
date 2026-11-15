# Final Report: Telangana PDS Shop Clustering

## 1. Key findings
- Portability (other-shop transactions) grew from 2.50M (Jan 2023) to 3.02M (May 2025),
  from 33.5% to 37.8% of all transactions. Growth flattened after late 2024.
- Total rice distributed rose about 28% while transactions rose about 7%, so rice per
  transaction increased. Likely cause: a change in ration scale (to be verified).
- 9,776 shop-month rows show more transactions than entitled cards, caused by portability.
- Transactions correlate 0.88 with entitled cards. Seasonality is weak in the available months.

## 2. Cluster profiles
| Persona | Shops | Avg trans/month | Portability | Volatility (CV) | Notes |
|---|---|---|---|---|---|
| Urban Portability Hubs | 1,338 | 923 | 76% | 0.13 | All distribute wheat, 92% in urban districts |
| Typical Rural Local Shops | 10,931 | 359 | 12% | 0.05 | 63% of all shops, very stable |
| Volatile Low-Volume Shops | 1,018 | 271 | 51% | 0.33 | Lowest rice per card (12), 11.5% DBSCAN noise |
| Semi-Urban Portability Shops | 3,964 | 596 | 53% | 0.08 | Highest rice per card (19.9), no wheat |

## 3. Model validation
- Silhouette: k=2 0.425, k=3 0.341, k=4 0.332, k=5 0.240. k=4 chosen as the last k before
  a sharp drop, and to give enough personas to profile.
- PCA: 2D explains 61.3% of variance, 3D 77.1%.
- Purity: 92% of Cluster 0 shops are in urban districts versus 7 to 15% for the others.

## 4. Recommendations
1. **Fraud prevention:** audit the 152 DBSCAN outliers first, starting with the 117 in the
   Volatile cluster. Alert on shops whose monthly volatility or utilization departs from
   their cluster by more than 3 standard deviations.
2. **Logistics:** treat Urban Portability Hubs as replenishment priorities. They serve
   mostly non-local cardholders, so stock should be sized to transactions, not to local cards.
3. **Policy impact:** portability is now structural, so allocations should follow actual
   footfall. Re-run clustering periodically to track movement between personas.
4. **Data:** extend to all months for a proper seasonality and volatility study.

## 5. Limitations
Five months only. Outliers are statistical flags, not proof of wrongdoing. The urban list
is approximate (Hyderabad, Medchal, Ranga Reddy, Warangal).
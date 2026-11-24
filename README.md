# Global Supply Chain Disruption Classification

Predicting whether a shipment will be **disrupted** before it leaves, using weather, geopolitical risk, carrier reliability and shipment details.

- **Kaggle notebook:** https://www.kaggle.com/code/amrbakr95/global-supply-chain-disruption-classification
- **Dataset:** [Global Supply Chain Risk & Logistics (2024-2026)](https://www.kaggle.com/datasets/nudratabbas/global-supply-chain-risk-and-logistics-2024-2026) (5,000 shipments, 14 columns, synthetic, CC0)

## Workflow

1. Data inspection: no missing values, no duplicates
2. Feature engineering: Month, year, day of week and quarter from the date; a `Route` column from origin and destination ports
3. EDA: barplots for categorical columns and histplots / boxplots / violinplots for numeric columns against the target
4. Log transform for the skewed `Lead_Time_Days` (skew 3.35 → 0.26), then dropped it to avoid **data leakage** (it is only known after the shipment arrives)
5. Split (80/20, stratified)
6. `ColumnTransformer`: StandardScaler for numeric columns, OneHotEncoder for categorical columns
7. `Pipeline` + `GridSearchCV` (5-fold, `roc_auc`) over 7 models with `class_weight='balanced'`
8. Threshold tuning to raise Recall
9. Final evaluation: classification report, confusion matrix, and analysis of missed disruptions

## Key EDA findings

| Feature | Effect on disruption |
|---|---|
| Weather_Condition | Strong (Hurricane 100%, Storm 80%, Clear 37%) |
| Geopolitical_Risk_Score | Strong (higher risk → more disruptions) |
| Carrier_Reliability_Score | Weak (about 66% → 56% as reliability increases) |
| Transport mode, product, route, distance, weight, fuel price, date | No clear effect |

## Model comparison (GridSearchCV)

| Model | CV ROC-AUC | Test ROC-AUC | Test Recall |
|---|---|---|---|
| XGBoost | 0.829 | 0.820 | 0.594 |
| Bagging | 0.828 | 0.829 | 0.690 |
| Random Forest | 0.828 | 0.822 | 0.581 |
| Logistic Regression | 0.827 | 0.829 | 0.680 |
| SVM | 0.822 | 0.819 | 0.628 |
| Decision Tree | 0.818 | 0.812 | 0.687 |
| KNN | 0.801 | 0.788 | 0.837 |

The top four models are practically tied. **Bagging** was chosen (best estimator: 100 Logistic Regression models, C=0.1, each trained on 50% of the data) because it has no overfitting and a good balance of metrics.

## Threshold tuning

Missing a disrupted shipment costs more than a false alarm, so the decision threshold was lowered from 0.50 to **0.40**:

| Threshold | Precision | Recall | Accuracy |
|---|---|---|---|
| 0.50 | 0.86 | 0.69 | 0.74 |
| 0.40 | 0.77 | 0.80 | 0.73 |

## Final result (test set, threshold 0.40)

- Caught **491 of 613** disrupted shipments (Recall = 0.80)
- 149 of 387 safe shipments were flagged as disrupted (false alarms)
- Most missed disruptions had calm weather and below-average geopolitical risk, so they were likely caused by factors not in the data. Improving the model would need new features rather than a more complex model.

## Tools

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, XGBoost

# IRRIGATION GENERALIZATION & SPATIAL TRANSFERABILITY REPORT

## The Nebraska Agronomic Anomaly
Nebraska produces over 1.6 billion bushels of corn annually, with approximately 55% of acreage receiving center-pivot groundwater irrigation from the Ogallala / High Plains Aquifer.
1. When standard machine learning models trained on rainfed Corn Belt states (IA, IL, IN, OH) are transferred to Nebraska without irrigation features, they exhibit a severe **underprediction bias of -8.92 bu/acre**.
2. Rainfed models misinterpret high July VPD and low rainfall in central/western Nebraska as agricultural drought, failing to recognize that groundwater applications fully decouple crop transpiration from rainfall deficits.
3. Incorporating the `vpd_irrigation_interaction` and `groundwater_ogallala_proxy` completely eliminates this spatial transfer penalty, reducing Nebraska test RMSE from 21.45 to **14.88 bu/acre** and bringing mean bias to **+0.18 bu/acre**.

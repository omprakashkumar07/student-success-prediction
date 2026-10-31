# Student Success Prediction

CA2 project for INT234 Predictive Analytics.

## Problem
Many students leave their course in between. In this project I try to predict whether a student will **dropout**, stay **enrolled** or **graduate**, using the data we have about the student and their first two semesters.

## Dataset
- Source: UCI Machine Learning Repository, "Predict Students' Dropout and Academic Success"
- Authors: Realinho et al. (2021), license CC BY 4.0
- 4424 students, 36 features, 3 classes (Graduate 2209, Dropout 1421, Enrolled 794)
- No missing values and no duplicate rows in the data

## What I did
1. Changed the target (Dropout, Enrolled, Graduate) into numbers.
2. Clipped the outliers in age, admission grade and previous qualification grade using IQR, so no rows were lost.
3. Made 2 new features: approval rate of 1st sem and 2nd sem (approved units / enrolled units).
4. Used one hot encoding on the code columns like course and application mode.
5. Did EDA with 5 graphs and checked correlation of every feature with dropout.
6. Split data 80/20 (stratified) and scaled it after the split.
7. Trained 5 models and compared them: Logistic Regression, KNN, Decision Tree, SVM, Random Forest.

## Results

| Model | Accuracy | Precision | Recall | F1 Score | AUC | Log Loss |
|---|---|---|---|---|---|---|
| Random Forest | 0.782 | 0.734 | 0.687 | 0.695 | 0.892 | 0.566 |
| Decision Tree | 0.755 | 0.700 | 0.677 | 0.685 | 0.841 | 1.387 |
| Logistic Regression | 0.766 | 0.705 | 0.675 | 0.682 | 0.877 | 0.600 |
| SVM | 0.750 | 0.687 | 0.646 | 0.653 | 0.871 | 0.601 |
| KNN | 0.631 | 0.557 | 0.528 | 0.532 | 0.759 | 2.896 |

- Random Forest gave the best result, but the top 3 models are very close in F1 score.
- Approval rate of 2nd sem has the strongest relation with dropout (-0.659).
- All models struggle with the Enrolled class (Random Forest recall is only 0.33), because it has the fewest students.
- KNN was the weakest, it does not work well with so many columns after encoding.

## Limitations
- Outlier limits and top-5 categories were calculated on the full data before the split, so there can be a small leakage.
- Most important features come from semester results, so this works as an early warning only after 2 semesters.

## How to run
1. Install libraries: `pip install pandas numpy matplotlib seaborn scikit-learn`
2. Keep `data.csv` inside the `Dataset` folder.
3. Change the `folder` path at the top of `student-success-prediction.py` to your own path.
4. Run `student-success-prediction.py`. Graphs are saved in `figures` and results in `model_results.csv`.

## Future work
- Try class weights or oversampling to improve the Enrolled class
- Tune the parameters of Random Forest and SVM
- Do the outlier clipping and category grouping after the split, to avoid leakage
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def create_visualizations():
    print("Loading data...")
    # Load dataset
    df = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")

    # Create visualizations directory
    os.makedirs("visualizations", exist_ok=True)

    print("Generating Target Distribution...")
    # 1. Target Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=df)
    plt.title('Target Variable (Label) Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.savefig('visualizations/target_distribution.png')
    plt.close()

    print("Generating Missing Values Plot (Train vs Test)...")
    # 2. Missing Values Visualization for Train and Test
    plt.figure(figsize=(12, 6))
    missing_train = (df.isnull().sum() / len(df)) * 100
    missing_test = (df_test.isnull().sum() / len(df_test)) * 100

    missing_df = pd.DataFrame({
        'Train % Missing': missing_train,
        'Test % Missing': missing_test
    })

    # Only plot columns that have missing values in either train or test
    missing_df = missing_df[(missing_df['Train % Missing'] > 0) | (missing_df['Test % Missing'] > 0)]

    if not missing_df.empty:
        missing_df.plot(kind='bar', figsize=(12, 6))
        plt.title('Missing Values Percentage: Train vs Test')
        plt.xlabel('Features')
        plt.ylabel('% Missing')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('visualizations/missing_values_comparison.png')
    plt.close()

    print("Generating Correlation Heatmap...")
    # 3. Correlation Heatmap for Numerical Features
    numerical_cols = ['emoticon_1', 'emoticon_2', 'emoticon_3', 'upvote', 'downvote', 'if_1', 'if_2']
    plt.figure(figsize=(10, 8))
    corr = df[numerical_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Numerical Features')
    plt.savefig('visualizations/correlation_heatmap.png')
    plt.close()

    print("Generating Comment Length Distribution...")
    # 4. Comment Length Distribution
    # Handle NaN comments before calculating length
    df['comment_length'] = df['comment'].fillna("").apply(len)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['comment_length'], bins=50, kde=True)
    plt.title('Distribution of Comment Lengths')
    plt.xlabel('Length of Comment (characters)')
    plt.ylabel('Frequency')
    plt.xlim(0, df['comment_length'].quantile(0.95)) # Cap at 95th percentile for better visualization
    plt.savefig('visualizations/comment_length_distribution.png')
    plt.close()

    print("Generating Scatter Plot...")
    # 5. Scatter Plot (Upvote vs Downvote)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='downvote', y='upvote', hue='label', palette='viridis', data=df, alpha=0.6)
    plt.title('Scatter Plot: Upvote vs Downvote by Label')
    plt.xlabel('Downvotes')
    plt.ylabel('Upvotes')
    # Use log scale if there's huge variance, otherwise stick to standard or limited limits
    plt.xlim(-1, df['downvote'].quantile(0.99))
    plt.ylim(-1, df['upvote'].quantile(0.99))
    plt.savefig('visualizations/scatter_upvote_vs_downvote.png')
    plt.close()

    print("Generating Box Plot...")
    # 6. Box Plot (Upvote by Label)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='label', y='upvote', data=df)
    plt.title('Box Plot: Upvotes across Labels')
    plt.xlabel('Label')
    plt.ylabel('Upvotes')
    plt.ylim(-1, df['upvote'].quantile(0.95)) # Limit outliers for better view
    plt.savefig('visualizations/boxplot_upvote_by_label.png')
    plt.close()

    print("Generating Outlier Detection Graphs...")
    # 7. Outlier Detection (Boxplots of Numerical Features)
    plt.figure(figsize=(12, 8))
    # Melting dataframe for easier seaborn plotting of multiple columns
    df_melted = pd.melt(df, value_vars=['upvote', 'downvote', 'if_1', 'if_2'])
    sns.boxplot(x='variable', y='value', data=df_melted)
    plt.title('Outlier Detection: Numerical Features')
    plt.xlabel('Feature')
    plt.ylabel('Value')
    plt.yscale('log') # Log scale because 'if_2' or upvotes can be heavily skewed with massive outliers
    plt.savefig('visualizations/outlier_detection_boxplots.png')
    plt.close()

    print("Generating Model Comparison and Cross Validation Graphs...")
    # 8. Model Comparison and CV Score Graphs

    # Baseline F1-Macro scores from notebook
    baseline_models = ['SGDClassifier', 'LogisticRegression', 'LinearSVC']
    baseline_scores = [0.7843, 0.8094, 0.7960]

    # Final tuned CV F1-Macro score
    final_model = 'Tuned LogReg (CV)'
    final_score = 0.8093

    models = baseline_models + [final_model]
    scores = baseline_scores + [final_score]

    # Plotting Model Comparison
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x=models, y=scores, palette=['lightblue', 'lightblue', 'lightblue', 'salmon'])
    plt.title('Model Comparison: Baseline vs Tuned (F1-Macro Score)')
    plt.xlabel('Models')
    plt.ylabel('F1-Macro Score')
    plt.ylim(0.75, 0.85) # Zoom in to see the differences

    # Add data labels
    for bar in bars.patches:
        bars.annotate(format(bar.get_height(), '.4f'),
                      (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      ha='center', va='center',
                      size=10, xytext=(0, 8),
                      textcoords='offset points')

    plt.tight_layout()
    plt.savefig('visualizations/model_comparison_cv_scores.png')
    plt.close()

    print("\n--- INSIGHTS ---")

    # Insights on class imbalance
    label_counts = df['label'].value_counts(normalize=True) * 100
    print(f"1. Class Imbalance: The dataset is highly imbalanced.")
    for label, pct in label_counts.items():
        print(f"   - Label {label}: {pct:.2f}%")

    # Insights on missing values
    print(f"\n2. Missing Values Analysis (Train vs Test):")
    if not missing_df.empty:
        for index, row in missing_df.iterrows():
            print(f"   - {index}: Train = {row['Train % Missing']:.2f}%, Test = {row['Test % Missing']:.2f}%")
    else:
        print("   - No missing values detected in train or test datasets.")

    # Correlation insights
    print("\n3. Feature Correlation:")
    print("   - Check visualizations/correlation_heatmap.png to see which numerical features are highly correlated.")

    # Outlier insights
    print("\n4. Outlier Analysis:")
    print("   - Boxplots show significant outliers in numerical features (upvote, downvote, if_1, if_2). Use log transformation or robust scalers to handle them.")

    # Model Comparison Insights
    print("\n5. Model Comparison (CV Scores):")
    print(f"   - Baseline Models (F1-Macro): {dict(zip(baseline_models, baseline_scores))}")
    print(f"   - Final Tuned Model (F1-Macro): {final_score}")
    print("   - Note: Logistic Regression baseline performed best. Tuning did not yield significant improvement over baseline LogReg, matching final CV score of ~0.8093.")

    print("\nVisualizations saved to the 'visualizations/' directory.")

if __name__ == "__main__":
    create_visualizations()

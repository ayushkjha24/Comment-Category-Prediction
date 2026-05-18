import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_visualizations():
    print("Loading data...")
    # Load dataset
    df = pd.read_csv("train.csv")

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

    print("Generating Missing Values Plot...")
    # 2. Missing Values Visualization
    plt.figure(figsize=(10, 6))
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if not missing_data.empty:
        sns.barplot(x=missing_data.index, y=missing_data.values)
        plt.title('Missing Values per Feature')
        plt.xlabel('Features')
        plt.ylabel('Missing Count')
        plt.xticks(rotation=45)
        plt.savefig('visualizations/missing_values.png')
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

    print("\n--- INSIGHTS ---")

    # Insights on class imbalance
    label_counts = df['label'].value_counts(normalize=True) * 100
    print(f"1. Class Imbalance: The dataset is highly imbalanced.")
    for label, pct in label_counts.items():
        print(f"   - Label {label}: {pct:.2f}%")

    # Insights on missing values
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    if not missing_pct.empty:
        print(f"\n2. Missing Values: The following columns have missing values:")
        for col, pct in missing_pct.items():
            print(f"   - {col}: {pct:.2f}% missing")
    else:
        print("\n2. Missing Values: No missing values detected in the dataset.")

    # Correlation insights
    print("\n3. Feature Correlation:")
    print("   - Check visualizations/correlation_heatmap.png to see which numerical features are highly correlated.")

    # Text length insights
    avg_len = df['comment_length'].mean()
    med_len = df['comment_length'].median()
    print(f"\n4. Text Characteristics:")
    print(f"   - Average comment length: {avg_len:.2f} characters")
    print(f"   - Median comment length: {med_len:.2f} characters")

    print("\nVisualizations saved to the 'visualizations/' directory.")

if __name__ == "__main__":
    create_visualizations()

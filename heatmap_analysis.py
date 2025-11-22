import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Load the data
print("Loading data...")
df = pd.read_csv('/Users/anastasiaperez/Downloads/institution-TJ/student-data/data_outcomes_with_tiers.csv')

print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Clean the data and filter for students with >4 point improvement
print("\nFiltering students with >4 point improvement...")
df_clean = df.dropna(subset=['Score Difference'])

# Filter for >4 point improvement
high_improvers = df_clean[df_clean['Score Difference'] > 4].copy()

print(f"Students with >4 point improvement: {len(high_improvers)}")
print(f"Score improvement range: {high_improvers['Score Difference'].min():.1f} to {high_improvers['Score Difference'].max():.1f} points")

# Analyze tier columns
tier_columns = ['Survey Tier', 'Large Group Tier', 'Small Group Tier', 'Class Participation Tier']
print(f"\nTier columns available: {tier_columns}")

# Create a summary of high improvers by tier
print("\nHigh improvers by tier:")
for col in tier_columns:
    if col in high_improvers.columns:
        tier_counts = high_improvers[col].value_counts()
        print(f"\n{col}:")
        for tier, count in tier_counts.items():
            print(f"  {tier}: {count} students")

# Create heat map data
# We'll create a matrix showing: Student ID vs Tier Type, colored by improvement score
print("\nPreparing heat map data...")

# Get unique student IDs and create a matrix
student_ids = high_improvers['student_id'].unique()
print(f"Number of unique high-improving students: {len(student_ids)}")

# Create a matrix: rows = students, columns = tier types
tier_matrix = []
student_labels = []
improvement_values = []

for student_id in sorted(student_ids)[:50]:  # Limit to top 50 for readability
    student_data = high_improvers[high_improvers['student_id'] == student_id].iloc[0]
    
    row = []
    for tier_col in tier_columns:
        if tier_col in student_data and pd.notna(student_data[tier_col]):
            # Convert tier to numeric: Tier 1 = 1, Tier 2 = 2, Tier 3 = 3
            tier_value = int(student_data[tier_col].split(' ')[-1]) if 'Tier' in str(student_data[tier_col]) else 0
            row.append(tier_value)
        else:
            row.append(0)  # Missing data
    
    tier_matrix.append(row)
    student_labels.append(f"Student {int(student_id)}")
    improvement_values.append(student_data['Score Difference'])

tier_matrix = np.array(tier_matrix)

# Create the heat map
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))

# Heat map 1: Tier levels
sns.heatmap(tier_matrix, 
            xticklabels=[col.replace(' Tier', '') for col in tier_columns],
            yticklabels=student_labels,
            annot=True, 
            cmap='RdYlBu_r',
            cbar_kws={'label': 'Tier Level (1=Best, 3=Needs Improvement)'},
            ax=ax1)
ax1.set_title('Student Tier Assignments\n(Students with >4 Point Improvement)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Tier Categories', fontsize=12)
ax1.set_ylabel('Students', fontsize=12)

# Heat map 2: Score improvements
improvement_matrix = np.array(improvement_values).reshape(-1, 1)
sns.heatmap(improvement_matrix,
            xticklabels=['Score Improvement'],
            yticklabels=student_labels,
            annot=True,
            fmt='.1f',
            cmap='Greens',
            cbar_kws={'label': 'MCAT Score Improvement (Points)'},
            ax=ax2)
ax2.set_title('MCAT Score Improvements\n(Students with >4 Point Improvement)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Metric', fontsize=12)
ax2.set_ylabel('Students', fontsize=12)

plt.tight_layout()
plt.savefig('/Users/anastasiaperez/Downloads/institution-TJ/high_improvers_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a combined analysis heat map
print("\nCreating combined analysis...")

# Create a correlation matrix between tiers and improvement
tier_improvement_data = []
for student_id in student_ids:
    student_data = high_improvers[high_improvers['student_id'] == student_id].iloc[0]
    
    row_data = {'student_id': student_id, 'improvement': student_data['Score Difference']}
    for tier_col in tier_columns:
        if tier_col in student_data and pd.notna(student_data[tier_col]):
            tier_value = int(student_data[tier_col].split(' ')[-1]) if 'Tier' in str(student_data[tier_col]) else np.nan
            row_data[tier_col] = tier_value
        else:
            row_data[tier_col] = np.nan
    
    tier_improvement_data.append(row_data)

tier_df = pd.DataFrame(tier_improvement_data)

# Create correlation analysis
print("\nCorrelation between tiers and improvement:")
correlations = {}
for tier_col in tier_columns:
    if tier_col in tier_df.columns:
        corr = tier_df[tier_col].corr(tier_df['improvement'])
        correlations[tier_col] = corr
        print(f"{tier_col}: {corr:.3f}")

# Summary statistics
print(f"\nSummary Statistics for High Improvers (>4 points):")
print(f"Total students: {len(high_improvers)}")
print(f"Average improvement: {high_improvers['Score Difference'].mean():.1f} points")
print(f"Median improvement: {high_improvers['Score Difference'].median():.1f} points")
print(f"Max improvement: {high_improvers['Score Difference'].max():.1f} points")

# Tier distribution analysis
print(f"\nTier Distribution Analysis:")
for tier_col in tier_columns:
    if tier_col in high_improvers.columns:
        print(f"\n{tier_col}:")
        tier_stats = high_improvers.groupby(tier_col)['Score Difference'].agg(['count', 'mean', 'std']).round(1)
        print(tier_stats)

print("\nHeat map saved as 'high_improvers_heatmap.png'")


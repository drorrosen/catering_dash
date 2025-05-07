import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("Enterprise Event & Catering Data Jan 2023 through February 2025(New- Events and Functions).csv",  encoding='latin1')
##### Remove missing values from columns with >0% and <30% missing
missing_pct = df.isna().mean()
cols_with_some_missing = missing_pct[(missing_pct > 0) & (missing_pct < 0.30)].index.tolist()
df = df.dropna(subset=cols_with_some_missing)

####### remove columns with +90% missing values
df.drop('Lkup_Space_Description', axis = 1 , inplace = True)


############# Ensure the column convert to lowercase
df['Event_Description'] = df['Event_Description'].astype(str).str.lower()

##############Coerce to numeric (parsable values stay, bad ones become NaN)
df['OrderedAttendance'] = pd.to_numeric(df['OrderedAttendance'], errors='coerce')
df['OrderedAttendance'] = df['OrderedAttendance'].astype(int)
df = df[df['OrderedAttendance'] >= 0]

############### date columns 
date_cols = ['StartDate', 'EndDate', 'FunctionStart', 'FunctionEnd']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')


##############convert object to numerical feat
numerical_cols = ['ActualRevenue', 'OrderedRevenue', 'ForecastRevenue', 'OrderedAttendance']
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    
    
######### identify all object-dtype columns. convert each to string and strip whitespace
object_cols = df.select_dtypes(include='object').columns.tolist()
for col in object_cols:
    df[col] = df[col].astype(str).str.strip()
    
######### Replace the literal "nan" string with a real missing value
cols_to_fix = ['Space', 'Space_Description', 'Event_Contact_Name']
for col in cols_to_fix:
    df[col] = df[col].replace('nan', pd.NA)
    
    
# ############## remove the least common categories from each categorical/oject feature
# remove_values = {
#     'Category_Code': ['9'],
#     'Category_Type': ['Burns, Madison'],
#     'Status': ['GWS'],
#     'Status_Description': ['Global Workplace Services (GWS)'],
#     'Type': ['4', 'N14B90B3'],
#     'EventType_Description': ['Internal Departmental Meeting', '3792'],
#     'Usage': ["Casey's", 'BayWa AG', 'City of New York Technical Leadership Team', 'Chipsoft'],
#     'BusinessGroup_Code': ['H01F5CCB'],
#     'BusinessGroup_Description': ['MMS'],
#     'BG_Desc_Masked': ['Eventions']
# }

# for col, bad_vals in remove_values.items():
#     non_null = df[col].notna()
#     temp = df.loc[non_null, col].astype(str).str.strip().str.lower()
#     bad = [v.lower() for v in bad_vals]
#     mask = temp.isin(bad)

#     # drop those rows
#     df = df.drop(df.loc[non_null].index[mask])

# #############remove outliers for numerical features
# Compute the 95th percentile for each and filter out values above it
for col in numerical_cols:
    q99 = df[col].quantile(0.99)
    df = df[df[col] <= q99]


################# filter out rows with StartDate before January 2023 or EndDate after June 2025
df['StartDate'] = pd.to_datetime(df['StartDate'], errors='coerce', utc=True)
df['EndDate']   = pd.to_datetime(df['EndDate'],   errors='coerce', utc=True)

start_min = pd.Timestamp('2023-01-01', tz='UTC')
start_max = pd.Timestamp('2025-02-01', tz='UTC')
end_max   = pd.Timestamp('2025-06-30', tz='UTC')

mask = ((df['StartDate'] >= start_min) &
#         (df['StartDate'] <= start_max) &
        (df['EndDate']   <= end_max))

df = df[mask].reset_index(drop=True)

df_first = df.drop_duplicates(subset='EventID', keep='first').reset_index(drop=True)


features = [
    'EventType_Description',
     'BusinessGroup_Description',
    'Category_Code',
    'Category_Type', 
    'Event_Description',
    'Status_Description',
    'BG_Desc_Masked',
]

top_n = 7

sns.set_style("whitegrid")
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(25, 15))
axes = axes.flatten()

for idx, feature in enumerate(features):
    ax = axes[idx]
    
    counts = df_first[feature].value_counts(normalize=True).head(top_n) * 100
    
    bars = counts.plot(
        kind='barh',
        ax=ax,
        color=plt.get_cmap('cividis')(np.linspace(0.2, 0.8, len(counts)))
    )
    
    ax.set_title(feature.replace('_', ' '), fontsize=16, fontweight='bold')
    ax.set_xlabel('Percentage (%)', fontsize=12)
    ax.set_ylabel('')

    ax.grid(True, linestyle='--', alpha=0.7, color='grey')
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    
    for bar in bars.patches:
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f'{bar.get_width():.2f}%',
            va='center',
            ha='left',
            fontsize=13
        )
    
# remove unused axes
for j in range(len(features), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


features = ['Function_Description','Usage_Description','Space_Description','Allocation']

top_n = 7

sns.set_style("whitegrid")
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(features):
    ax = axes[idx]

    counts = df[feature].value_counts(normalize=True).head(top_n) * 100

    bars = counts.plot(
        kind='barh',
        ax=ax,
        color=plt.get_cmap('cividis')(np.linspace(0.2, 0.8, len(counts)))
    )

    ax.set_title(feature.replace('_', ' '), fontsize=16, fontweight='bold')
    ax.set_xlabel('Percentage (%)', fontsize=12)
    ax.set_ylabel('')
    
    ax.grid(True, linestyle='--', alpha=0.7, color='grey')
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)

    for bar in bars.patches:
    ax.text(
            bar.get_width() + 0.5,
        bar.get_y() + bar.get_height() / 2,
            f'{bar.get_width():.2f}%',
        va='center',
        ha='left',
            fontsize=13
        )
    
# remove unused axes
for j in range(len(features), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


columns = ['ActualRevenue', 'OrderedRevenue', 'ForecastRevenue']

# 2. Styling
sns.set_style("whitegrid")
colors = sns.color_palette("plasma", len(columns))

# 3. Create subplots: 1 row, 3 columns
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
axes = axes.flatten()

# 4. Plot each histogram with a two-line title, using feature names with spaces
for ax, col, color in zip(axes, columns, colors):
    data = df_first[col].dropna()
    mean, std, summ = data.mean(), data.std(), data.sum()
    feature_name = col.replace('_', ' ')

    ax.hist(data, bins=30, color=color)
    ax.set_title(
        f"{feature_name}\n(Mean={mean:.2f}, Std={std:.2f}\nSum={summ:.2f})",
        fontsize=14, fontweight='bold'
    )

    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel(feature_name, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7, color='grey')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

# 5. Final layout
plt.tight_layout()
plt.show()


sns.set_style("whitegrid")


# 2. Select the single column
col = 'OrderedAttendance'
data = df[col].dropna()

# 3. Compute stats
mean = data.mean()
std  = data.std()
summ = data.sum()
feature_name = 'Ordered Attendance'

# 4. Create one plot
fig, ax = plt.subplots(figsize=(8, 4))

# 5. Plot histogram
color = sns.color_palette("plasma", 1)[0]
ax.hist(data, bins=30, color=color)

# 6. Two-line title with stats
ax.set_title(
    f"{feature_name}\nMean={mean:.2f}, Std={std:.2f}, Sum={summ:.2f}",
    fontsize=14, fontweight='bold'
)

# 7. Labels and styling
ax.set_xlabel('Frequency', fontsize=12)
ax.set_ylabel(feature_name, fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6, color='grey')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)

plt.tight_layout()
plt.show()



print('df.shape:', df.shape)
print('\n')
print('Number of Unique events: ', df['EventID'].nunique())
print('\n')
print('StartDate Min', df['StartDate'].min())
print('EndDate Min', df['EndDate'].max())
print('\n')
x = df.isnull().mean()
print('Percentage of missing values: \n', x[x>0])

# 1. Collapse the data to 1 row per EventID (take first values)
event_level_df = df.groupby('EventID').agg({
    'StartDate': 'first',
    'ActualRevenue': 'first',
#     'OrderedRevenue': 'first',
#     'ForecastRevenue': 'first',
    'Category_Type': 'first',             # Optional if you want forecast by Category later
    'EventType_Description': 'first'      # Optional
}).reset_index()

# 2. Create a Month column
event_level_df['Month'] = event_level_df['StartDate'].dt.to_period('M')

# 3. Group by Month
revenue_per_month = (
    event_level_df.groupby('Month')[['ActualRevenue']]
    .sum()
    .reset_index()
)

# Convert Month to datetime for plotting and modeling
revenue_per_month['Month'] = revenue_per_month['Month'].dt.to_timestamp()
revenue_per_month = revenue_per_month[revenue_per_month['Month'] <= pd.Timestamp('2025-02-01')]

# 4. Plot total actual revenue
plt.figure(figsize=(14,6))
plt.plot(revenue_per_month['Month'], revenue_per_month['ActualRevenue'], marker='o', label='Actual Revenue')
# plt.plot(revenue_per_month['Month'], revenue_per_month['OrderedRevenue'], linestyle='--', label='Ordered Revenue')
# plt.plot(revenue_per_month['Month'], revenue_per_month['ForecastRevenue'], linestyle=':', label='Forecast Revenue')

plt.title('Monthly Actual Revenue Trend', fontsize=18, fontweight='bold')
plt.xlabel('Month', fontsize=14)
plt.ylabel('Revenue ($)', fontsize=14)
# plt.legend(fontsize=13)
plt.grid(True, linestyle='--', alpha=0.6, color='grey')
# plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 2️⃣  Calculate duration in days (inclusive)
df['Duration_Days'] = ((df['EndDate'] - df['StartDate']).dt.days + 1)

df_dur = df.groupby('EventID').agg({
    'EventType_Description': 'first',
    'Duration_Days': 'first'
}).reset_index()

# 1️⃣  optional – keep only event types with ≥10 records
ct = df_dur['EventType_Description'].value_counts()
keep_types = ct[ct >= 10].index
plot_df = df_dur[df_dur['EventType_Description'].isin(keep_types)].copy()

# 2️⃣  order categories by median duration (longest → shortest)
order = (plot_df.groupby('EventType_Description')['Duration_Days']
               .median()
               .sort_values(ascending=False)
               .index)

# 3️⃣  box‑plot
plt.figure(figsize=(12, 6))
sns.boxplot(data=plot_df,
            x='EventType_Description',
            y='Duration_Days',
            order=order,
            showfliers=False, palette='Set3')

plt.title('Duration (Days) by Event Type', fontsize=15, fontweight='bold')
plt.xlabel('Event Type')
plt.ylabel('Duration (Days)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

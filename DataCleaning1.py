import pandas as pd


df = pd.read_csv(r"C:\Users\Nikhil Kulkarni\Downloads\New folder\injuries_2010-2020.csv")


df = df.drop(columns=['Acquired'])


df = df.rename(columns={'Relinquished': 'Player_Name'})


for col in ['Player_Name', 'Team', 'Notes']:
    df[col] = df[col].astype(str).str.strip().str.title()


df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Year'] = df['Date'].dt.year


df = df.dropna(subset=['Player_Name', 'Team'])


df = df.drop_duplicates(subset=['Player_Name', 'Date'])


import re

def get_injury_type(note):
    note = note.lower()
    # Add/modify keywords as needed for your dataset!
    for keyword in ['fracture', 'sprain', 'tear', 'torn', 'strain', 'break', 'contusion', 'surgery', 'dislocation', 'bruise', 'soreness', 'procedure', 'inflammation', 'rupture']:
        if keyword in note:
            return keyword.title()
    return 'Other'

df['Injury_Type'] = df['Notes'].apply(get_injury_type)

# Reset index
df = df.reset_index(drop=True)

print("Rows after cleaning:", len(df))
print("Unique players:", df['Player_Name'].nunique())
print("Unique teams:", df['Team'].nunique())
print("Injury years:", sorted(df['Year'].unique()))
print("\nTop 10 players by number of injuries:")
print(df['Player_Name'].value_counts().head(10))
print("\nTop 10 teams by number of injuries:")
print(df['Team'].value_counts().head(10))
print("\nInjuries per year:")
print(df['Year'].value_counts().sort_index())
print("\nMost common injury types:")
print(df['Injury_Type'].value_counts())


print("\nSample cleaned data:")
print(df.head())

df.to_csv(r"C:\Users\Nikhil Kulkarni\Downloads\New folder\injuries_2010-2020_cleaned.csv", index=False)

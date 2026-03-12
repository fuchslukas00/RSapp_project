import pandas as pd

df = pd.read_csv(r"C:\Users\lukas\Documents\Studium\Remote_Sensing_Products\project\RSapp_project\Analysis\merged_csv_ger\model_table_ww_2017_2023_ger.csv")

print(df.isna().sum()[df.isna().sum() > 0].sort_values(ascending=False))
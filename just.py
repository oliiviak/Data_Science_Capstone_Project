df.to_csv('/Users/zumiis/final_not4git/for github/interest_rate_only.csv', index=False)

path = '/Users/zumiis/final_not4git/for github/data_with_interest_rate_all.csv'
#path2 = '/Users/zumiis/final_not4git/for github/data_with_interest_rate_gdp_all.csv'

df_with_gdp = data_combine(path2, df_gdp)
print(df_with_gdp.shape) #4997 rows 
print(df_with_gdp.isna().sum())

df_with_gdp.to_csv('/Users/zumiis/final_not4git/for github/data_with_interest_rate_gdp_all.csv', index=False)

df_with_gdp

df_nonans = cleaning_if_nans(df_clean, 209)
print(df_nonans.isna().sum())
df_nonans #240 rows 
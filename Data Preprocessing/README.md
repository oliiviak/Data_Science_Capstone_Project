The objective of the preprocessing section is to create a dataset that contains all the data we will need for the multivariate model. Values for each independent variable, for each country will be appended to the main dataset. For each independent variable, there is a notebook outlining the preprocessing steps to create a dataframe that only contains values for that particular variable. Once that dataframe has been constructed, it will be appended to the main dataset. The order in which the variables are appended does not matter, however, for further iteration steps, it is important to remember to append to the new main dataset as opposed to the original main dataset (that only contains exchange rate data). For example, the order in which the notebooks were written is interest rate, GDP growth rate, terms of trade, unemployment. Thus, our final dataset will be completed once the unemployment data has been appended. When saving the new main dataset, it depends on individual preference whether to save to and overwrite the same dataset every time, or create a new dataset for each iteration. The notebooks have been written and formatted according to the latter option. 

The independent variables included are the following - interest rate, GDP growth rate, terms of trade, and unemployment.

----------------------------------------------------------------------------------------------------------------------------

Throughout the project files, 'main dataset' will always be referring to <a href="https://www.kaggle.com/brunotly/foreign-exchange-rates-per-dollar-20002019?select=Foreign_Exchange_Rates.csv" target="_blank">this</a> dataset from Kaggle. 


The data for interest rates was downloaded from <a href="https://fred.stlouisfed.org/searchresults/?st=Immediate%20Rates%3A%20Less%20than%2024%20Hours%3A%20Call%20Money%2FInterbank%20Rate" target="_blank">FRED</a> using "Immediate Rates: Less than 24 Hours: Call Money/Interbank Rate".


The data for GDP growth rate was downloaded from <a href="https://fred.stlouisfed.org/searchresults/?st=Gross%20Domestic%20Product%20by%20Expenditure%20in%20Constant%20Prices%3A%20Total%20Gross%20Domestic%20Product%20for%20" target="_blank">FRED</a> using "Gross Domestic Product by Expenditure in Constant Prices: Total Gross Domestic Product" with the format set to "Quarterly, Growth Rate Previous Period, Seasonally Adjusted".


Since GDP growth rate data for China from the FRED website is only available at an annual frequency, it was downloaded from <a href="https://data.imf.org/?sk=388dfa60-1d26-4ade-b505-a05a558d9a42" target="_blank">IMF</a> using "GDP and Components" and the "Gross Domestic Product, Deflator" values.


The exports and imports data used for calculating terms of trade was downloaded from <a href="https://data.imf.org/?sk=388dfa60-1d26-4ade-b505-a05a558d9a42" target="_blank">IMF</a> using "Data by Country/Economy" and "Goods, Value of Exports, US Dollars" for exports and "Goods, Value of Imports, CIF, US Dollars" for most imports. 

- Ideally, "Goods, Value of Imports, FOB, US Dollars" would have been used in order to match the valuation of exports, however, too many countries did not have any records for FOB imports, as such, CIF imports were used, except for Australia & Mexico, wherein the terms of trade for both countries was calculated using FOB import values as there were NaNs in their CIF import values.

The data for the Unemployment Rate was downloaded from <a href="https://fred.stlouisfed.org/searchresults/?st=Harmonized%20Unemployment%20Rate%3A%20Total%3A%20All%20Persons" target="_blank">FRED</a> using "Harmonized Unemployment Rate: Total: All Persons".

- It should be noted that there is no unemployment data from FRED that is in line with the format for China, Indonesia, South Africa, and Switzerland. As such, for these countries, multivariate analysis is to be conducted without taking the unemployment rate into consideration. 

----------------------------------------------------------------------------------------------------------------------------

Also note that when renaming the variables, it is imperative that the country code naming convenctions are consistent. 
If other naming conventions are to be used, then the variable names in the main dataset must be changed and the country code values must be altered accordingly. 

Here are the country code naming conventions used throughout this project. 

- Australia -- AUD
- New Zealand  -- NZD
- UK -- GBP
- Brazil -- BRL
- Canada -- CND
- China -- CNY
- Indonesia -- IDR
- South Korea -- KRW
- Mexico -- MXN
- South Africa -- ZAR
- Denmark -- DKK
- Japan -- JPY
- Norway -- NOK
- Sweden -- SEK
- Switzerland -- CHF
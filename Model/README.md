This section contains the functions file and the notebook that outlines the basic linear strucutre of the multivariate LSTM model. The final dataset is also included. 
(Note: the final dataset presented here is different from the one created in '04_load_data_unemployment' in that it includes two dummy variables (for the financial crisis and H1N1 years) and the price of gold in USD.)

----------------------------------------------------------------------------------------------------------------------------

The data for gold prices was downloaded from <a href="https://www.gold.org/goldhub" target="_blank">here</a>, and you are required to sign up with an account in order to access historical data.

Full credit goes to <a href="https://www.youtube.com/watch?v=gSYiKKoREFI&ab_channel=Dr.VytautasBielinskas" target="_blank">this</a> tutorial on how to structure a multivariate LSTM model and <a href="https://heartbeat.fritz.ai/using-a-keras-long-shortterm-memory-lstm-model-to-predict-stock-prices-a08c9f69aa74" target="_blank">this</a> article for the base LSTM model composition that gave the best results. 

----------------------------------------------------------------------------------------------------------------------------

This model was optimized with Canadian data in mind, so while it still does perform acceptably well for other countries, there are some noticeable discrepancies where the predictions end and where the forecasts begin. However, these discrepancies only apply with a forecast range that is very long (in this case, the standard of 365 days), when forecasted on smaller ranges (e.g. 30 days as shown in the notebook), this discrepancy does not present itself. Due to the original time constraint of two weeks for completing this project, these results are the "good enough" status for presentation. As such further developments include a multitude of possibilities from developing a model that operates on a rolling window/online machine learning system wherein the short range forecasted values are appended to the final dataset, the model is retrained on the updated dataset, and made to return forecasts for the next iteration of the short range time interval. Alternatively, a more 'powerful' model that makes use of more robust independent variables or an alternative model composition could be considered. 


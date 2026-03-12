# Run this once in your terminal to check spellings
import pandas as pd
df = pd.read_csv('C:\\Users\\semwa\\OneDrive\\Desktop\\Supply-Chain-Risk-Engine\\data\\DataCoSupplyChainDataset.csv', encoding='ISO-8859-1')
print(df['Shipping Mode'].unique())
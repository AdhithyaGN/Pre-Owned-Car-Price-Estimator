import pandas as pd
df=pd.read_excel('notebook\True_Value_data.xlsx')
df.info()

feature columns=['POC Sales Date',
 'Sale Type',
 'Customer City',
 'Model',
 'Vehicle Sold Category',
 'YOM',
 'Date Of Registration',
 'Selling Mileage',
 'Buying Price',
 'Total Actual RF',
 'Warranty Charges',
 'Insurance Charges',
 'Vehicle Sell Price',
 'Finance/Cash']

sale type=['Retail To Customer', 'Wholesale To Broker', 'Scrap']

customercity=['ERNAKULAM', 'KOTTAYAM', 'PALA', 'Thodupuzha ', 'PATHANAMTHITTA',
       'IDUKKI', 'KOLLAM', 'CHANGANASERRY', 'KANJIRAPALLY',
       'MUVATTUPUZHA', 'THIRUVALLA', 'KOZHIKODE', 'MALAPPURAM',
       'ALAPPUZHA', 'COCHIN', 'KANJIRAPPALLY', 'CHENGANNUR',
       'KOTTARAKKARA', 'THRISSUR', 'VALANCHERRY', 'KATTAPPANA',
       'ALLEPPEY', 'KOTHAMANGALAM', 'CHERTHALA', 'KAYAMKULAM', 'WAYANAD',
       'Erattupetta ', 'KASARGOD', 'MAVELLIKARA', 'ADOOR', 'PALAKKAD',
       'Vaikom ', 'CALICUT', 'TRIVANDRUM', 'CANNANORE ', 'Athirampuzha ',
       'QUILON', 'MALLAPPALLY', 'THODUPUZHA', 'FORT KOCHI', 'MAVELIKARA',
       'ANGAMALLY', 'THIRUVANANTHAPURAM', 'RANNI', 'PANDALAM', 'KONNI',
       'ALAKODE', 'AYOOR', 'Pallichal ', 'CHANGANASSERY']


Model=['RITZ', 'SWIFT DZIRE', 'ALTO 800', 'M 800', 'ZEN', 'ALTO', '-',
       'SWIFT', 'WAGON R', 'A-STAR', 'ZEN ESTILO', 'OMNI', 'EECO',
       'CELERIO', 'ALTO K10', 'ERTIGA', 'NEW ERTIGA', 'VITARA BREZZA',
       'NEW SWIFT', 'BALENO', 'CIAZ', 'ALTO K10 (NEW)', 'DZIRE', 'WagonR',
       'Ignis', 'SCROSS', 'SX4', 'S-PRESSO', 'HYUNDAI', 'HONDA',
       'NEW CELERIO', 'TATA', 'NEW BALENO', 'NISSAN', 'OTHERS', 'RENAULT',
       'TOYOTA', 'VOLKSWAGEN', 'MAHINDRA', 'CHEVROLET', 'EICHER', 'FIAT',
       'XL6', 'FORD', 'FORCE MOTORS', 'ESTEEM', 'SUPER CARRY',
       'NEW ALTO K10']

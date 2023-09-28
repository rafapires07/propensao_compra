import pickle
import inflection
import pandas as pd


class Propensao( object ):
    def __init__( self ):
        self.home_path='/Users/Rafael/Desktop/Programacao/repos/propensao_compra/pkl/'
        self.age_scaler = pickle.load( open( self.home_path + 'age_scaler.pkl', 'rb') )
        self.annual_premium_scaler = pickle.load( open( self.home_path + 'annual_premium_scaler.pkl', 'rb') )
        self.region_code_encoder = pickle.load( open( self.home_path + 'region_code_encoder.pkl', 'rb') )
        self.sales_channel_encoder = pickle.load( open( self.home_path + 'sales_channel_encoder.pkl.pkl', 'rb') )
        self.vehicle_damage_encoder = pickle.load( open( self.home_path + 'vehicle_damage_encoder.pkl', 'rb') )
        self.vintage_scaler = pickle.load( open( self.home_path + 'vintage_scaler.pkl', 'rb') )
        
    def data_cleaning( self, df1 ):
        #Renomear Colunas
        old_cols = ['id', 'Gender', 'Age', 'Driving_License', 'Region_Code',
       'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium',
       'Policy_Sales_Channel', 'Vintage']
        snakecase = lambda x:inflection.underscore(x)
        new_cols = list(map(snakecase, old_cols))
        df1.columns = new_cols
        
        #Convertendo data
        #region_code para int
        df1['region_code'] = df1['region_code'].astype(int)

        #policy_sales_channel para int
        df1['policy_sales_channel'] = df1['policy_sales_channel'].astype(int)
        
        return df1
    
    def data_preparation( self, df3 ):
        # Standardization
        #annual_premium
        df3['annual_premium'] = annual_premium_scaler.transform( df3[['annual_premium']].values )
        
        # Rescaling
        #age
        df3['age'] = age_scaler.transform( df3[['age']].values )
        
        #vintage
        df3['vintage'] = vintage_scaler.transform( df3[['vintage']].values )
       
        # Encoding
        #gender - one hot encoding
        df3 = pd.get_dummies(df3,prefix='gender',columns=['gender'])
        
        #region_code - target encoder
        df3['region_code'] = region_code_encoder.transform(df3['region_code'])
        
        #vehicle_age - Ordinal encoder
        mapping = {"< 1 Year": 1,  "1-2 Year": 2, "> 2 Years": 3}
        df3['vehicle_age'] = df3['vehicle_age'].map(mapping)
        
        #vehicle_damage - Label Encoder
        df3['vehicle_damage'] = vehicle_damage_encoder.transform( df3['vehicle_damage'].values)
        
        #policy_sales_channel - Target encoder
        df3['policy_sales_channel'] = sales_channel_encoder.transform(df3['policy_sales_channel'])
        
        fs_cols = ['vintage', 'annual_premium', 'age', 'region_code',
           'policy_sales_channel', 'vehicle_damage', 'previously_insured']
        
        return df3[ fs_cols ]
    
    def get_prediction( self, model, original_data, test_data ):
        # Previsão
        pred = model.predict_proba( test_data )
        
        # Colocando a previsão no dataframe original
        original_data['score'] = pred[:, 1].tolist()
        
        return original_data.to_json( orient='records', date_format='iso' )
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import pickle
import regex
import gzip
from sklearn.model_selection import GridSearchCV
st.set_page_config(page_title="Vehicle Price Predictor", layout="wide")
st.title('Australian Vehicle Price Predictor')
with open('firse.pkl', 'rb') as model_file:
    model,feature_names = pickle.load(model_file)
with gzip.open('compressed_Pickle.pkl', 'rb') as model_file:
    model= pickle.load(model_file)
fea={
    'Brand':['Ssangyong','MG', 'BMW' ,'Mercedes-Benz' ,'Renault' ,'Land' ,'Nissan' ,'Toyota','Honda' ,'Volkswagen' ,'Ford' ,'Mitsubishi' ,'Subaru' ,'Hyundai' ,'Jeep',
 'Volvo' ,'Mazda','Abarth','Holden','Audi','Kia','Mini','Suzuki','Porsche',
 'Peugeot','Isuzu','Lexus','Jaguar','Rolls-Royce','Skoda','Fiat','Haval',
 'Citroen','LDV','HSV','Foton','Mahindra','Maserati','GWM','Ram','Tesla',
 'Alfa','Genesis','Dodge','Chrysler','Great','Opel','Bentley','Ferrari',
 'Cupra','Chevrolet','Lamborghini','FPV','McLaren','Iveco','Chery',
 'Infiniti','BYD','Tata','Aston','Daewoo','Saab','Proton','Smart'],
 'Model':['Rexton','MG3','430I','E500','Arkana','Rover','Pulsar','86','Jazz',
 'HiAce','Golf','X3','118D','Fiesta','Outlander','Amarok','Outback',
 'Mirage','Camry','I45','Territory','Qashqai','Tucson','Focus','X-Trail',
 'Corolla','Yaris','Patriot','S60','Triton','E250','A250','CX-5','Falcon',
 '595','CR-V','I30','6','Kluger','Santa','Cascada','ML320','Lancer','A5',
 'RIO','Cherokee','Tiguan','ASX','220I','3','Imax','XV','Cooper',
 'Commodore','I20','BT-50','CX-7','Renegade','Grand','Impreza','X4','118I',
 'Q7','Celerio','Liberty','C-HR','Forester','CX-3','City','Cruze','RAV4',
 'CLC200','Caddy','Civic','Compass','Ranger','2','M135I','330I','Q3',
 'Accent','Sportage','Cayenne','Cayman','Navara','Hilux','Venue','X1',
 'Veloster','M4','Swift','Transit','308','Trax','Captiva','Cerato','RS3',
 'C200','D-MAX','A200','Puma','Sorento','MU-X','IS350','Stinger',
 'Landcruiser','A3','Trafic','208','HR-V','A4','Accord','SQ5','IS300','XF',
 'Polo','ZS','M140I','Q2','218I','Patrol','Passat','Megane','Ghost',
 'Crafter','Tarago','Iload','Rodeo','320D','Astra','Colorado','Octavia',
 'UX200','Rapid','Picanto','Carnival','ZST','Calais','500','GLA250','H9',
 'V60','Stonic','RX300','Pajero','Vitara','CX-9','Clubman','C4','2008',
 'Acadia','Barina','C220','C250','Mustang','Elantra','Macan','Kona','X5',
 'Pathfinder','E2000','C43','320I','S3','M5','WRX','Dualis','RX350','IX35',
 'Viva','GLC43','Commander','Almera','C30','Multivan','R8','GR','Koleos',
 'G10','Touareg','Escape','Q5','IS250C','Ioniq','116I','GLC300','SQ7',
 'Eclipse','Mondeo','RX350L','Clubsport','Hiace','Aurion','T60',
 'Transporter','Celica','Tunland','PIK-UP','420I','Kamiq','RX450H','GTS',
 'CLA250','Ghibli','Fabia','Optima','328I','XC60','Z4','Tank','M2','A7',
 'F-Pace','ML350','Juke','A1','Echo','C63','X6','1500','C300','Model',
 'GLE63','Sonata','Romeo','B180','Statesman','Altima','Superb','LS460',
 'I40','ORA','225I','Everest','Kangoo','CLA45','ES250','A45','E200',
 'Sprinter','RX450HL','A6','330D','Caprice','V40','GV80','Niro','Staria',
 'BRZ','Odyssey','FJ','Countryman','Journey','T-Cross','ES350','300','3D',
 'Wrangler','NX300H','GS450H','Baleno','T-ROC','508','Ignis','NX300',
 '3008','V80','CLS250','XC90','GLA45','RCZ','CLA','M240I','HS','GLC',
 'Wall','380','Ecosport','RS4','GLE','M340I','MPV','Haval','Taycan','ML63',
 'Combo','RS','GLA35','Sebring','428I','M235I','E350','Rondo','Karoq',
 'B200','Convertible','Fortuner','435I','230I','Maxima','E55','DS3','TT',
 'E400','DS4','EOS','Captur','Kuga','GLC250','Arteon','Getz','MX-5',
 'Kodiaq','A8','320CI','Grancabrio','S40','Festiva','Express','UTE',
 'CX-30','NX250','Jetta','4008','RS6','CT','Mulsanne','Trailblazer',
 'Challenger','Freemont','IS300H','Panda','Gladiator','ML400','B2500',
 '500X','CLS53','Micra','Clio','360','Scirocco','GLA180','G10+',
 'Formentor','XC40','ES300H','XE','X2','Gtsr','NX200T','Silverado','5',
 'GV70','Levante','Urus','SX4','X7','S450','ML250','Quattroporte','NX350H',
 'Viano','323','218D','135I','F6','IS250','Xjsc','Endura','A180','CLK320',
 'M3','Monaro','MX-30','GLE350','Leaf','720S','Crewman','GL500','Seltos',
 'E-Pace','Ateca','S350','CX-8','RS5','GLC63','Fairmont','540C','Coupe',
 'Fairlane','GLC200','Mifa','F-Type','CLA200','LX570','G63','LC500',
 'Master','Levorg','Granvia','Scala','911','E63','S65','Palisade','Jimny',
 'Boxster','GLE400','S5','XL-7','Q8','650S','S4','F250','Expert','M8','X',
 'LX500D','128TI','740I','V250','Continental','Nitro','RC350','SL55',
 'E-Tron','RX-8','Tribeca','Equinox','Jackaroo','R300','220D','RX270',
 'Prius','Laguna','Alto','X250','Murano','125I','Tiida','F430','Avensis',
 'Panamera','Valente','SLC','EQA','GLE300','XUV500','S-Cross','228I','GLA',
 'MB100','Vito','D90','RX330','Tribute','Musso','718','Magna','A190',
 'Caravelle','RC200T','E300','E240','Integra','Z','GTC4','Granturismo',
 'Terracan','Daily','GT-R','M6','Courier','Steed','S63','Roomster','318I',
 'C180','SQ2','Korando','5008','E220','Kizashi','GLC220','GLC350',
 'Pursuit','AMG','F','RX500H','V','Marco','C5','C3','G350','CLS500',
 'Beetle','Senator','Stavic','I4','Trajet','H6','C230','307','Doblo',
 '335I','300C','H8','Adventra','LX600','S450L','330CI','CR-Z','323I',
 'Cabrio','IS200T','CLK63','CT200H','Berlingo','Omoda5','Supra','H2',
 'CLS350','California','GLS350','RC','Vectra','Genesis','ES300','GL63',
 'G70','GLS500','K2700','SC430','Q50','Paceman','GLA200','Avalon','Atto',
 '520D','Corsa','EQB','XUV700','G80','340I','Spectra','A35','GS350','370Z',
 'GLE450','GLA220','GLS','B250','EQS53','EQC','Actyon','C350','Malibu',
 'Maloo','XKR','XC70','GL320','CLK500','407','Soul','570S','Aventador',
 'XJ8','207','530I','CC','612','120D','E53','EV6','740LI','NX350','Arnage',
 'Q60','IX','525I','420D','ML500','121','Maybach','LX470','Maverick',
 'Xenon','SL350','APV','Martin','GLB','730D','CLA35','X350','I-Pace','5D',
 'Lanos','120I','Verada','R320','Avalanche','GLE250','PT','J11','CLK280',
 '316I','C40','3-Sep','Laser','GS250','LX450D','Prius-C','Coaster','ML300',
 'Grange','S16','Scudo','Kadjar','530D','350Z','325I','I3','GR86','GLE43',
 'Punto','SQ8','V50','Fortwo','Epica','GT','M440I','560','Born','A170',
 'SL500','300ZX','Sierra','HDT','Leon','SLK350','S430','420','B4000',
 'Scorpio-N','Corvette','190'],
 'UsedOrNew':['DEMO','USED','NEW'],
 "Transmission":['Automatic','Manual'],
 "DriveType":['AWD','Front','Rear','4WD'],
 "FuelType" :['Diesel','Premium','Unleaded','Hybrid','LPG','Leaded'],
 "BodyType":['SUV','Hatchback','Coupe','Commercial','Sedan','Ute / Tray',
 'People Mover','Convertible','Wagon','Other']
}
def predict(input_data):
    m=model.predict(input_data)
    return m

selected_features={}
for feature,options in fea.items():
    selected_features[feature]=st.selectbox(f'Select {feature}',options)
year=st.number_input('year',value=0)
Engine=st.text_input('Engine (x cyl,y L)',placeholder='4 cyl, 2.2 L',value=' 0 cyl, 0.0 L')
FuelConsumption=st.text_input('FuelConsumption',placeholder='8.7 L',value='0')
Kilometres=st.number_input('Kilometres',value=0)
CylindersinEngine=st.text_input('CylindersinEngine',value=0)
Doors=st.number_input('Doors',value=0)
Seats=st.number_input('Seats',value=0)
FuelConsumption= FuelConsumption.replace(r'[^0-9]', '')
data={
    'Brand':selected_features['Brand'],
    'Year':year,
    "Engine":Engine,
    'Model':selected_features['Model'],
    'UsedOrNew':selected_features['UsedOrNew'],
    "Transmission":selected_features['Transmission'],
    "DriveType":selected_features['DriveType'],
    'FuelType':selected_features['FuelType'],
    'FuelConsumption':FuelConsumption,
    'Kilometres':Kilometres,
    'CylindersinEngine':CylindersinEngine,
    'BodyType':selected_features['BodyType'],
    'Doors':Doors,
    'Seats':Seats
}
st.write(selected_features)
def load_and_transform(feature, new_value):
    with open(f'{feature}_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    
    # Transform the new user input
    transformed_value = encoder.transform(new_value)
    
    return transformed_value

encoded_user_data = {}
for feature, value in selected_features.items():
    data[feature] = load_and_transform(feature,[value])
st.write(data)    

df=pd.DataFrame([data],columns=['Brand', 'Year','Engine', 'Model', 'UsedOrNew', 'Transmission', 'DriveType',
       'FuelType', 'FuelConsumption', 'Kilometres', 'CylindersinEngine',
       'BodyType', 'Doors', 'Seats'])

df['cylinders']=df['CylindersinEngine']
df['engine_displacement'] = df['Engine'].str.extract(r'(\d+\.\d+|\d +L)').astype(float)
df['FuelConsumption'] = df['FuelConsumption'].str.extract(r'([0-9.]+)').astype(float)
st.write(df)
df=df[feature_names] 
st.write(df.dtypes)
if st.button('submit'):
    x=predict(df)
    st.write(f"The predicted price is {x[0]}")







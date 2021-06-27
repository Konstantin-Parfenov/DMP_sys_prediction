#!/opt/anaconda/envs/bd9/bin/python
import sys
import pandas as pd

import numpy as np
import pandas as pd
import os, sys
from datetime import datetime

import json

import urllib.parse
from urllib.parse import unquote
from urllib.parse import urlparse
from keras.models import load_model
import time

#CONSTANTS

PREDICTION_FEATURES_BORDER = 3
TRAINING_FEATURES_BORDER = 4



start_time = time.time()
# your script


columns=['gender','age','uid','user_json']
print("Reading of raw data...", end =" ")
theRawData = pd.read_table(
    sys.stdin, 
    sep='\t', 
    header=None, 
    names=columns
)
print('OK')
elapsed_time = time.time() - start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
# Здесь необходимо преобразовать модель точно так-же как и в обучающей выборке

#Процедура. Фильтрует домен из url
def toDomain( url ):
    if url.startswith('http://http') : url = url[7:]
    if url.startswith('http://&referrer=') : url = url[17:]
        
    parsed_url = urlparse( urllib.parse.unquote( url ).strip() )
    if parsed_url.scheme not in ['http','https']: return None

    url = parsed_url.netloc.strip()

    if url.startswith('www.') : url = url[4:]

    dpoint = url.rfind(':')     
    if dpoint != -1 : url = url[:dpoint]    

    dpoint = url.find('&')     
    if dpoint != -1 : url = url[:dpoint]    

    dpoint = url.rfind('@')     
    if dpoint != -1 : url = url[dpoint+1:]    
       
    return url if url.rfind('.') != -1 else None

#Процедура разбирает JSON и возвращет домен и timestamp
def workupDomain( szJsonCollection ):
    return [[ toDomain(value['url']), value['timestamp']]  for value in json.loads( szJsonCollection )['visits']]

#Перебираем элементы, сохраняя из данных тока домен и timestamp
print("Перебираем элементы, сохраняя из данных тока домен и timestamp", end =" ")
theRawData['domain'] = theRawData['user_json'].apply( workupDomain )
theRawData.drop(['user_json'], axis=1, inplace=True)
print('OK')
elapsed_time = time.time() - start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

#Загужаем theRadar данные
print("Загужаем theRadar данные", end =" ")


theRadar = pd.read_csv('/data/home/konstantin.parfenov/project/subjectByDomen.csv')  

print('OK')
elapsed_time = time.time() - start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


#Создаем датасет footprints
print("Создаем датасет footprints", end =" ")

global uid
uid = []

global gender
gender = []

global age
age = []

global url 
domain = []

global footprint
footprint = []

#Процедура. Пробераем по всем элементам json и формируем из них отдельные строки (для каждого листа)
def extractDomain(row):
    for rowDomain in row['domain'] :
        uid.append( row['uid'] )
        gender.append( row['gender'] )
        age.append( row['age'] )
        domain.append( rowDomain[0] )
        footprint.append( rowDomain[1] )
    
#Вызываем обработку каждой строки.Рузультат не сохраняем, он в глобальных списках
theRawData.apply( extractDomain, axis = 1 )
#Строем DataFrame для доменов
theFootprints = pd.DataFrame ({'uid':uid, 'gender' : gender,'age' : age, 'domain' : domain, 'footprint': footprint  }) 
theFootprints.info()

def domainToSubject( szDomain ):
    if szDomain is None : return 'unknown.patch'
    
    theSplit = szDomain.split('.')
    return szDomain if len(theSplit) < 3 else theSplit[-2] + '.' + theSplit[-1]

theFootprints['key'] = theFootprints['domain'].apply( domainToSubject )
theShort = theRadar[['domain', 'subject', 'subject1']]
theShort.rename({'domain':'key'}, axis=1, inplace=True )
theShort = theShort.append({'key': 'unknown.patch', 'subject':'неизвестный'},  ignore_index=True)

theFootprints = pd.merge(theFootprints, theShort, on='key', how = 'left')
theFootprints['subject'].fillna( 'прочее', inplace=True )
theFootprints['subject1'].fillna( 'прочее', inplace=True )
theFootprints.head()

print('OK')
elapsed_time = time.time() - start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


print("Создаем основной массив данных под модель. Наименование df_test", end =" ")
theFeatch = theRawData.copy()
theFeatch.drop(['domain'], axis=1, inplace=True)
theFeatch.head()

#Я из дропа убрал Hour of a day, я на предыдущем шаге не стал его формировать
theFootprints.drop(['gender','age','domain'], axis=1, inplace=True)
theFootprints = theFootprints.groupby(['uid', 'subject', 'subject1'], as_index = False ).count()
theFootprints.set_index(['uid'], inplace=True)
theFootprints.head()

def toFetch( uid ):
    theUid = theFootprints.loc[uid]

    dict_ = {'uid':uid }
    if  isinstance(theUid, pd.DataFrame) :
        for _, theLine in theUid.iterrows():
            if theUid['subject'] is not None :
                dict_[theLine['subject'] + ' ' + theLine['subject1'] ] = theLine['footprint'] 
    else :
        if theUid['subject'] is not None :
            dict_ [theUid['subject'] + ' ' + theUid['subject1'] ] =  theUid['footprint']
    return dict_

theTmp =  pd.DataFrame( [toFetch( uid ) for uid in theRawData.uid] )

#!!!!Здесь я переименовал датафрейм df_test
theFeatch = pd.merge(theFeatch, theTmp, on='uid')
df_test = theFeatch.copy()
print('OK')
elapsed_time = time.time() - start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))



print(df_test.shape)
df_test.head()
'''
данный кусок кода не нужен поскольку мы не обучаем модель а делаем предикт

print('Создание доплонительных столбцов данных...', end=" ")
#Создадим пол-возраст в начале датафрейма
df_main.insert(loc=1, column='gender_age', value = df_main['gender'] + df_main['age'])
df_main['gender_age'] = df_main['gender_age'].map({'F18-24': 0, F'25-34': 1, 'F35-44': 2,'F45-54': 3, 'F>=55': 4,'M18-24': 5, 'M25-34': 6, 'M35-44': 7,'M45-54': 8, 'M>=55': 9 })


print('OK')
'''
#Отделяем фичи от датасета
print('Отделяем фичи от датасета...', end=" ")

#Создадим дополнительные переменные для разделения колонок датасета на фичи 
#   !!!! ПОДСТАВЛЯЕМ КОНФИГ ИЗ ВЕРХА СКРИПТА !!!!
column_headers = list(df_test.columns.values)
feature_border = PREDICTION_FEATURES_BORDER
feature_culumn_headers = list(df_test.columns.values)[feature_border:]
features_len = len(column_headers[feature_border:])



print('OK количество фич: ',features_len)
elapsed_time = time.time() - start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


#заменяем NaN фичи на 0

print('Заменяем NaN фичи на 0...', end=" ")

df_test[column_headers[feature_border:]]=df_test[column_headers[feature_border:]].fillna(value=0)

print('OK')
df_test.head(10)
elapsed_time = time.time() - start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))



#Загрузка модели из файла /data/home/konstantin.parfenov/model_gender_age.h5
print('Загрузка модели из файла /data/home/konstantin.parfenov/model_gender_age.h5', end=" ")
# load model
model_gender_age = load_model('/data/home/konstantin.parfenov/model_gender_age.h5')
# summarize model.
model_gender_age.summary()
print('OK')
elapsed_time = time.time() - start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


#Создание предикта и предикт_пробы
print('Создание предикта и предикт пробы', end=" ")

X_test=df_test[column_headers[feature_border:]]

Y_test=model_gender_age.predict_classes(X_test)
Y_test_proba=model_gender_age.predict(X_test)

print('OK')
elapsed_time = time.time() - start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))



#Создание столбцов age и gender на основании предикта в формате пандаса
print('Создание столбцов age и gender на основании предикта в формате пандаса', end=" ")

dict_gender = {0:'F', 1:'F', 2:'F',3:'F', 4:'F', 5:'M', 6:'M', 7:'M', 8:'M', 9:'M'}
dict_age = {0:'18-24', 1:'25-34', 2:'35-44', 3:'45-54', 4:'>=55', 5:'18-24', 6:'25-34', 7:'35-44', 8:'45-54', 9:'>=55',}

gender_predict = pd.Series([dict_gender.get(n, n) for n in Y_test])
age_predict = pd.Series([dict_age.get(n, n) for n in Y_test])

print('OK')
elapsed_time = time.time() - start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

#Сбор output датафрейма
print('Сбор output датафрейма', end=" ")
frame = { 'uid': df_test.iloc[:,PREDICTION_FEATURES_BORDER], 'age': age_predict, 'gender': gender_predict } 
output = pd.DataFrame(frame) 
print('OK')
elapsed_time = time.time() - start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


#output в json
print('output в json', end=" ")
output = output[['uid', 'gender', 'age']]
output.sort_values(by='uid',axis = 0, ascending = True, inplace = True)
sys.stdout.write(output.to_json(orient='records'))
print('OK')
elapsed_time = time.time() - start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
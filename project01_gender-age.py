#!/opt/anaconda/envs/bd9/bin/python
import numpy as np
import pandas as pd
import os, sys

import json

import urllib.parse
from urllib.parse import unquote
from urllib.parse import urlparse

pd.options.mode.chained_assignment = None  # default='warn'

szYandexRadarPath = 'subject.csv' #Путь к файлу категорий

global theRadar #Глобальный справочник категорий доменов

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
       
    if url.rfind('.') == -1: return None

    theSplit = url.split('.')
    return url if len(theSplit) < 3 else theSplit[-2] + '.' + theSplit[-1]

def toSubject(domain):
    try: return theRadar.loc[domain]['subject'].strip()  
    except: return 'прочее'
    
#Генерируем фичи для uid
def toFetch( row ):
    try:
        theJson = json.loads( row[1].user_json)
        theUidList = [ toDomain(value['url']) for value in  theJson['visits'] ]
        theUidList = [value for value in theUidList if value is not None]
    except : 
        theUidList = ['nodata.ru']    

    theUid = pd.DataFrame( 0, index = theUidList, columns = ['footprint'] )
    theUid.index.name = 'domain'
    theUid = theUid.groupby(['domain']).count()
    theUid['subject'] = theUid.apply(lambda value : toSubject(value.name), axis = 1 )

    dict_ = {'uid':row[1].uid }
    for _, theLine in theUid.iterrows():
        dict_[ theLine['subject'] ] = theLine['footprint'] 
    return dict_

def workupLoad() :
    #Читаем справочник Yandex Radar    
    theRadar = pd.read_csv( szYandexRadarPath, sep=',' )
    theRadar.set_index(['domain'], inplace=True)

    #Читает данные из входного потока
    theColumnsList=['gender','age','uid','user_json']
    theRawData = pd.read_table( sys.stdin, sep='\t', header=None, names= theColumnsList )
    return theRawData

def workupPrepare(theInputFrame) :
    # Здесь необходимо преобразовать модель точно так-же как и в обучающей выборке
    # подаем на вход загруженные данные, забираем обратно фичи

    # Так-же необходимо отфильтровать пользователей с уже известным AGE и SEX
    # Я заметил в задании фразу !должны быть только те пользователи, у которых пол и возрастная категория изначально неизвестны!
    #TODO

    #Перебираем элементы, сохраняя из данных тока категории домена 
    theFeature =  pd.DataFrame( [toFetch( row ) for row in theInputFrame.iterrows()] )
    theFeature.set_index(['uid'], inplace=True)
    return theFeature

def workupSave(theResultFrame):
    theResultFrame.reset_index(inplace=True)
    theResultFrame = theResultFrame[['uid', 'gender', 'age']]
    theResultFrame.sort_values(by='uid', axis = 0, ascending = True, inplace = True)

    sys.stdout.write( theResultFrame.to_json( orient='records' ) )


def workup() :
    ##########################################################################
    #1. Загружаем данные из входного потока
    ##########################################################################
    theInputFrame = workupLoad()

    ##########################################################################
    #2. Преобразуем входные данных
    ##########################################################################
    theFeature = workupPrepare(theInputFrame)

    ##########################################################################
    #3. Загружаем предобученную модель
    ##########################################################################
    #считать модель из файла в переменную vectorizer
    #Путь к модели необходимо указывать относительно вашей домашней директории.
    #Например, если в директории name.surname есть папка project01, в которой лежит модель project01_model.pickle,
    #то путь к модели в скрипте будет выглядеть как project01/project01_model.pickle.
    '''
    import pickle

    model_file = "project01_model.pickle"
    p1 = pickle.load(open(model_file, 'rb'))
    '''

    ##########################################################################
    #4. Обрабатываем подготовденные данные с помощью предобученной модели
    ##########################################################################
    #Запускае обработку данных с помощью загруженной модели
    #заменим пустые значения в gender и age на 'M' и '25-34'
    theResultFrame = theFeature

    theResultFrame['gender'] = 'M'
    theResultFrame['age'] = '25-34'

    ##########################################################################
    #4. Сохраняем результаты обработки в json формате в выходной поток
    ##########################################################################
    workupSave(theResultFrame)


if __name__ == '__main__':
    workup()  
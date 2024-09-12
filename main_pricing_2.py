# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:12:51 2024

@author: carlo
"""
import json
import pandas as pd
import numpy as np
import gspread
from datetime import datetime as dt
import cleaner_functions as clean_func

import streamlit as st

def get_GP(qsupp, qsp):
    '''
    Computes gross profit wrt supplier price (qsupp)
    '''
    try:
        #return round(100*(float(qsp)/float(qsupp) - 1), 2
        return round(100*(1- (float(qsupp)/float(qsp))), 2)
    except:
        return np.nan

def ceil_5(n):
    '''
    Rounds off to nearest highest number ending in 5.
    '''
    return np.ceil(n/5)*5

def consider_GP(value, GP):
    '''
    Incorporate required GP in price to ceil_5
    
    Parameters:
    ----------
        value : int or float
            given price
        GP : float
            required gross profit in decimal (< 100)
    '''
    return ceil_5(value*(1+float(GP)/100))

def promotize(value, GP_promo):
    '''
    Apply promo calculation to fixed price
    '''
    return consider_GP(value*4/3, float(GP_promo))

def query_gulong_data() -> pd.DataFrame:
    '''
    Query main gulong data from Redash
    '''
    
    # http://app.redash.licagroup.ph/queries/131
    url1 = "http://app.redash.licagroup.ph/api/queries/131/results.csv?api_key=FqpOO9ePYQhAXrtdqsXSt2ZahnUZ2XCh3ooFogzY"
    # query
    df_data = pd.read_csv(url1, parse_dates=[
                          'supplier_price_date_updated', 
                          'product_price_date_updated'],
                           date_format = '%m/%d/%y %H:%M')
    # select columns
    df_data = df_data[['make', 'model', 'section_width', 'aspect_ratio', 'rim_size', 
                       'pattern', 'load_rating', 'speed_rating', 'stock', 'name', 
                       'cost', 'srp', 'promo', 'mp_price', 'b2b_price', 
                       'supplier_price_date_updated', 'product_price_date_updated', 
                       'supplier_id', 'sale_tag', 'product_id', 'activity']]
    # rename 
    df_data = df_data.rename(columns = {'name' : 'supplier',
                                        'cost' : 'supplier_price',
                                        'srp' : 'GulongPH_slashed',
                                        'promo' : 'GulongPH',
                                        'mp_price' : 'marketplace',
                                        'b2b_price' : 'b2b',
                                        'supplier_price_date_updated' : 'supplier_updated',
                                        'product_price_date_updated' : 'gulong_updated',
                                        })
    
    # import gulong makes list
    makes_list = clean_func.import_makes()

    # cleaning
    # 1st pass
    df_data = df_data[df_data.section_width.notna() & df_data.aspect_ratio.notna() &\
                      df_data.rim_size.notna() & df_data.product_id.notna() &\
                          (df_data.make.notna() & (df_data.make != ' '))]
    df_data.loc[:, 'make'] = df_data.apply(lambda x: clean_func.clean_make(
        x['make'], makes_list, model=x['model']), axis=1)
    df_data.loc[:, 'section_width'] = df_data.apply(
        lambda x: clean_func.clean_width(x['section_width']), axis=1)
    df_data.loc[:, 'aspect_ratio'] = df_data.apply(
        lambda x: clean_func.clean_aspect_ratio(x['aspect_ratio'], model=x['model']), axis=1)
    df_data.loc[:, 'rim_size'] = df_data.apply(
        lambda x: clean_func.clean_diameter(x['rim_size']), axis=1)
    df_data.loc[:, 'speed_rating'] = df_data.apply(
        lambda x: clean_func.clean_speed_rating(x['speed_rating']), axis=1)
    df_data.loc[:, 'correct_specs'] = df_data.apply(lambda x: clean_func.combine_specs(
        x['section_width'], x['aspect_ratio'], x['rim_size'], mode='MATCH'), axis=1)

    # 2nd pass
    df_data = df_data[df_data.section_width.notna() & df_data.aspect_ratio.notna() &\
                      df_data.rim_size.notna() & df_data.product_id.notna() &\
                          (df_data.make.notna() & (df_data.make != ' '))]
    df_data.loc[:, 'model_'] = df_data.loc[:, 'model']
    
    df_data.loc[:, 'model'] = df_data.apply(lambda x: clean_func.combine_sku(x['make'],
                                                                             x['section_width'],
                                                                             x['aspect_ratio'],
                                                                             x['rim_size'],
                                                                             x['pattern'],
                                                                             x['load_rating'],
                                                                             x['speed_rating']), axis=1)
    df_data = df_data.drop_duplicates(subset = ['model', 'product_id', 
                                                'load_rating', 'speed_rating'], 
                                      keep = 'first')
    
    return df_data

def set_supplier_df(df_data : pd.DataFrame) -> pd.DataFrame:
    '''
    Construct supplier dataframe from redash gulong data query
    
    Parameters:
    -----------
        - df_data : pd.DataFrame
            redash gulong data query
    
    Returns:
    --------
        - df_supplier : pd.DataFrame
            resulting supplier dataframe
    
    '''
    
    df_supplier = df_data[['model','supplier',
                           'supplier_price',
                           'supplier_updated']].copy().sort_values(by='supplier_updated', ascending=False)
    df_supplier = df_supplier.drop_duplicates(subset=['model','supplier'], keep='first')
    df_supplier = df_supplier.groupby(['model','supplier'], group_keys=False).agg(price = ('supplier_price', lambda x: x))
    df_supplier = df_supplier.unstack('supplier').reset_index().set_index(['model'])
    # set supplier name as column name format: ('price', supplier name)
    df_supplier.columns = [i[1] for i in df_supplier.columns] 
    df_supplier['supplier_max_price'] = df_supplier.fillna(0).max(axis=1)
    # reset index ('model') as column
    df_supplier = df_supplier.reset_index()
    
    return df_supplier

def import_competitor_data():
    '''
    Import scraped competitor data from BQ
    
    Returns:
    --------
        - df_competitor : pd.DataFrame
        - latest_sheet : str
            sheet name of sheet to be used from gsheet/date in str format
    
    '''
    # import scraped competitor data
    try:
        creds = st.secrets['secrets']
    except:
        with open('secrets.json') as file:
            creds = json.load(file)
    
    # open gsheet
    gsheet_key = "12jCVn8EQyxXC3UuQyiRjeKsA88YsFUuVUD3_5PILA2c"
    gc = gspread.service_account_from_dict(creds)
    sh = gc.open_by_key(gsheet_key)
    
    sheet_list = []
    worksheet_list = sh.worksheets()
    for item in range(len(worksheet_list)):
      if 'Copy' not in worksheet_list[item].title:
          sheet_list.append(worksheet_list[item].title)
    # select latest sheet
    latest_sheet = max(sheet_list)
    worksheet = sh.worksheet(latest_sheet)
    # convert to dataframe
    df_competitor = pd.DataFrame(worksheet.get_all_records())
    df_competitor = df_competitor[['sku_name','price_gogulong',
                                   'price_tiremanila', 
                                   'price_partspro', 
                                   'qty_tiremanila', 'year']]
    df_competitor.columns = ['model', 'GoGulong','TireManila', 'PartsPro', 
                             'qty_tiremanila', 'year']
    with pd.option_context("future.no_silent_downcasting", True):
        df_competitor = df_competitor.replace('', np.nan).infer_objects(copy = False)
    df_competitor['GoGulong_slashed'] = df_competitor['GoGulong'].apply(lambda x: float(x)/0.8)
    
    return df_competitor, latest_sheet


def acquire_data() -> dict:
    '''
    Wrapper function for gathering data from redash, competitors
    
    Parameters:
    -----------
        None.
    
    Returns:
    --------
        - output : dict
            dictionary containing final dataframe, columns, competitor 
            dataframe, and latest sheet used for the date 

    '''
    # redash query
    df_data = query_gulong_data()
    # construct supplier dataframe
    df_supplier = set_supplier_df(df_data)
    # gulong dataframe
    df_gulong = df_data.drop(columns = ['supplier', 
                                        'supplier_price',
                                        'supplier_updated'],
                             axis = 1).copy().sort_values(by='gulong_updated',
                                                          ascending=False)
    df_gulong = df_gulong.drop_duplicates(subset='model',
                                          keep='first').drop('gulong_updated',
                                                             axis = 1)
    # import competitor data
    df_competitor, latest_sheet = import_competitor_data()
    # merge dataframes
    df_temp = df_gulong.merge(df_supplier, 
                              on = 'model', 
                              how= 'outer').merge(df_competitor, 
                                                 on='model', 
                                                 how='left').sort_values(by= 'model')
    df_temp = df_temp.dropna(subset = 'supplier_max_price')
    # df_temp['dimensions'] = df_temp.apply(lambda x: '/'.join(x[['section_width',
    #                                                             'aspect_ratio',
    #                                                             'rim_size']].astype(str)),axis=1)
    df_temp['dimensions'] = df_temp.apply(lambda x: clean_func.combine_specs(x['section_width'], 
                                                                             x['aspect_ratio'], 
                                                                             x['rim_size'], 
                                                                             mode = 'MATCH'),
                                          axis = 1)
    
    # calculate gogulong GP
    df_temp['GoGulong_GP'] = df_temp.loc[:,['supplier_max_price',
                                            'GoGulong']].apply(lambda x: round(get_GP(x['supplier_max_price'],
                                                                                      x['GoGulong']),2),
                                                                                      axis=1)
    df_temp = df_temp.loc[df_temp['GulongPH'] != 0]
    df_temp['GulongPH_GP'] = df_temp.loc[:,['supplier_max_price',
                                            'GulongPH']].apply(lambda x: round(get_GP(x['supplier_max_price'],
                                                                                      x['GulongPH']),2),
                                                                                      axis=1)
                                                                                      
    cols_option = ['GoGulong_slashed', 'GoGulong_GP', 
                   'GulongPH_GP'] + list(df_supplier.columns)
    
    df_temp['3+1_promo_per_tire_GP25'] = df_temp['supplier_max_price'].apply(lambda x: promotize(x,25))
    df_temp = df_temp.drop_duplicates(subset='model', keep='first')
    
    if 'model' in cols_option:
        cols_option.remove('model')
    if 'supplier_max_price' in cols_option:
        cols_option.remove('supplier_max_price')
    
    output = {'df_final' : df_temp,
              'cols_option' : cols_option,
              'df_competitor' : df_competitor,
              'backend_last_update' : dt.today().date().strftime('%Y-%m-%d'),
              'comp_last_update' : latest_sheet}
    
    return output


def adjust_wrt_gogulong(df : pd.DataFrame, 
                        GP_15 : int = 15,
                        GP_20a : int = 5,
                        GP_20b : int = 1, 
                        b2b : int = 25,
                        affiliate : int = 27,
                        mp : int = 25) -> pd.DataFrame:
    
    '''
    Adjusts GulongPH prices with respect to scraped GoGulong prices
    
    Parameters:
    -----------
        - df : pd.DataFrame
            main dataframe (df_final from acquire_data() after implement_sale)
    
    '''
    
    df_adj = df.loc[df['GulongPH'] > df['GoGulong']].copy()
    if len(df_adj)==0:
        return df_adj
    
    #when gogulong GP <= 15%, set gulong GP to 15%
    df_adj.loc[df_adj['GoGulong_GP']<15,'GulongPH'] = df_adj.loc[df_adj['GoGulong_GP']<15,'supplier_max_price'].apply(lambda x: consider_GP(x,GP_15))
  
    #when gogulong GP <= 20%, set gulong price = gogulong price (floor_5)
    df_adj.loc[df_adj['GoGulong_GP'].between(15, 20, inclusive = 'left'), 'GulongPH'] = df_adj.loc[df_adj['GoGulong_GP'].between(
        15, 20, inclusive = 'left'), 'GoGulong'].apply(lambda x: ceil_5(x-GP_20a))

    #when gogulong GP >20%, set gulong price GP =  gogulong price GP - 1
    df_adj.loc[df_adj['GoGulong_GP'] >= 20, 'GulongPH'] = df_adj.loc[df_adj['GoGulong_GP'] >= 20, :].apply(
        lambda x: consider_GP(x['supplier_max_price'], 
                              x['GoGulong_GP']-GP_20b), 
                                axis=1)  # math.ceil(x['GoGulong_GP']-GP_20b))

    df_adj.loc[:,'GulongPH_slashed'] = df_adj.loc[:,'supplier_max_price'].apply(lambda x: consider_GP(x,30))
    df_adj.loc[:,'b2b'] = df_adj.loc[:,'supplier_max_price'].apply(lambda x: consider_GP(x,b2b))
    df_adj.loc[:,'affiliate'] = df_adj.loc[:,'supplier_max_price'].apply(lambda x:  consider_GP(x,affiliate))
    df_adj.loc[:,'marketplace'] = df_adj.loc[:,'supplier_max_price'].apply(lambda x:  consider_GP(x,mp))
    return df_adj

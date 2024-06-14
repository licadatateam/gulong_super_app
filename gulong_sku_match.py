# -*- coding: utf-8 -*-
"""
Created on Mon May 27 00:02:33 2024

@author: carlo
"""
import pandas as pd
import numpy as np
import streamlit as st
import json
import gspread
import re
from fuzzywuzzy import fuzz, process

import cleaner_functions as clean_func

#url = 'https://docs.google.com/spreadsheets/d/1WWPj-iEoCoTm4y97iHpRjEKuqsJYrgBznBGHmyDk2Xo/edit#gid=0'

# import scraped competitor data

def import_credentials():
    try:
        creds = st.secrets['secrets']
    except:
        with open('secrets.json') as file:
            creds = json.load(file)
    return creds

#@st.cache_data
def import_sheet(url, creds):
    
    # open gsheet
    gsheet_key = re.search('(?<=spreadsheets/d/).*(?=/edit)', url)[0]
    #gsheet_key = "1WWPj-iEoCoTm4y97iHpRjEKuqsJYrgBznBGHmyDk2Xo"
    gc = gspread.service_account_from_dict(creds)
    sh = gc.open_by_key(gsheet_key)
    
    # open worksheet
    working_sheet = sh.worksheet(sh.worksheets()[0].title)
    # convert worksheet to  dataframe
    pre_df = pd.DataFrame.from_records(working_sheet.get_all_records()).replace('', np.nan)
    
    return pre_df

@st.cache_data
def get_gulong_data() -> pd.DataFrame:
    '''
    Get gulong.ph data from backend
    
    Returns
    -------
    df : dataframe
        Gulong.ph product info dataframe
    '''
    show_cols = ['sku_name', 'raw_specs', 'price_gulong', 'name', 'brand', 
                 'width', 'aspect_ratio', 'diameter', 'correct_specs', 
                 'load_rating', 'speed_rating', 'product_id', 'activity']
    
    try:
        ## 1. Import from redash query api key
        # http://app.redash.licagroup.ph/queries/131
        url1 =  "http://app.redash.licagroup.ph/api/queries/131/results.csv?api_key=FqpOO9ePYQhAXrtdqsXSt2ZahnUZ2XCh3ooFogzY"
        
        df = pd.read_csv(url1, 
                         parse_dates = ['supplier_price_date_updated',
                                        'product_price_date_updated'],
                         date_format = '%m/%d/%y %H:%M')
              
        ## 2. rename columns
        df = df.rename(columns={'model': 'sku_name',
                                'name': 'supplier',
                                'pattern' : 'name',
                                'make' : 'brand',
                                'section_width' : 'width', 
                                'rim_size':'diameter', 
                                'promo' : 'price_gulong',
                                'activity' : 'active'}).reset_index(drop = True)
        
        ## 3. Perform data filtering and cleaning
        df.loc[df['sale_tag']==0, 'price_gulong'] = df.loc[df['sale_tag']==0, 'srp']
        # 1st pass
        df = df[df.width.notna() & df.aspect_ratio.notna() & df.diameter.notna() &\
                df.product_id.notna() & (df.brand.notna() & (df.brand != ' '))]
        df.loc[:, 'width'] = df.apply(lambda x: clean_func.clean_width(x['width']), axis=1)
        df.loc[:, 'aspect_ratio'] = df.apply(lambda x: clean_func.clean_aspect_ratio(x['aspect_ratio']), axis=1)    
        df.loc[:, 'diameter'] = df.apply(lambda x: clean_func.clean_diameter(x['diameter']), axis=1)
        # 2nd pass
        df = df[df.width.notna() & df.aspect_ratio.notna() & df.diameter.notna() &\
                df.product_id.notna() & (df.brand.notna() & (df.brand != ' '))]    
        df.loc[:, 'raw_specs'] = df.apply(lambda x: clean_func.combine_specs(x['width'], x['aspect_ratio'], x['diameter'], mode = 'SKU'), axis=1)
        df.loc[:, 'correct_specs'] = df.apply(lambda x: clean_func.combine_specs(x['width'], x['aspect_ratio'], x['diameter'], mode = 'MATCH'), axis=1)
        df.loc[:, 'name'] = df.apply(lambda x: clean_func.fix_names(x['name']), axis=1)
        df.loc[:, 'sku_name'] = df.apply(lambda x: clean_func.combine_sku(str(x['brand']), 
                                                               str(x['width']),
                                                               str(x['aspect_ratio']),
                                                               str(x['diameter']),
                                                               str(x['name']), 
                                                               str(x['load_rating']), 
                                                               str(x['speed_rating'])), 
                                                               axis=1)
        df['load_rating'] = df['load_rating'].astype('str')
        df = df[df.name != '-']
        df = df.drop_duplicates(subset = ['sku_name', 'product_id', 'load_rating', 'speed_rating'], 
                                keep = 'first')
        
    except Exception as e:
        raise e
    
    return df[show_cols]

# find matching pattern name from df_gulong
def name_match(s, ref, 
               scorer = fuzz.WRatio, 
               cutoff = 95,
               with_brand : bool = True):
    
    try:
        s = clean_func.fix_names(s)
        
        match = process.extractOne(s, ref.pattern, scorer = scorer, 
                           score_cutoff = cutoff)
        if match:
            result = match[0]
            try:
                brand = ref.loc[match[-1], 'make']
            except:
                brand = None
        else:
            result = None
            brand = None
    except:
        result = None
        brand = None
    
    if with_brand:
        return pd.Series([result, brand])
    else:
        return result

def clean_data(df, df_gulong):
    
    df_data = df.copy()
    
    # remove unnecessary rows
    df_data = df_data.dropna(subset = ['section_width', 'aspect_ratio', 'rim_size', 
                                 'pattern'], how = 'all')
    df_data = df_data.rename(columns = {'activity' : 'active'})
    
    makes_list = clean_func.import_makes()
    df_data.loc[:, 'brand'] = df_data.apply(lambda x: clean_func.clean_make(
        x['brand'], makes_list, model=x['pattern']), axis=1)
    
    df_data[['section_width', 
             'aspect_ratio', 
             'rim_size']] = df_data.apply(lambda x: pd.Series(clean_func.clean_tire_size(x['tire_sku'])), 
                                          axis=1)
    
    df_data['correct_specs'] = df_data.apply(lambda x: clean_func.combine_specs(x['section_width'],
                                                                                x['aspect_ratio'],
                                                                                x['rim_size'],
                                                                                'MATCH'),
                                                                     axis = 1)
    
    # determine matching similar model names from reference
    df_data['similar_pattern'] = df_data.apply(lambda x: name_match(x['pattern'], 
                                                                 df_gulong[df_gulong.pattern.notna()], 
                                                                 with_brand = False),
                                                             axis = 1)
    
    df_data['load_rating'] = df_data['load_rating'].astype('str')
    
    return df_data

def match_df(df1, df2):
    '''
    Merge dataframes 
    
    '''
    show_cols = ['product_id', 'model', 'GulongPH', 'activity']
    left_cols = ['similar_pattern', 'correct_specs',
                 'speed_rating','brand']
    right_cols = ['pattern', 'correct_specs',
                  'speed_rating', 'make']
    
    
    # merge
    merged = df1.merge(df2[show_cols + right_cols], 
                       left_on = left_cols,
                       right_on = right_cols,
                       how = 'left')
    merged = merged.rename(columns = {'pattern_x' : 'pattern'})
    merged = merged.drop(labels = ['pattern_y'], axis=1)
    merged = merged.drop_duplicates(subset = ['brand', 'tire_sku',
                                              'pattern', 'LIST PRICE'],
                                    keep = 'first')
    merged = merged.dropna(subset = ['brand'])
    merged = merged[merged.brand != 'BRAND']
    
    return merged

def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

if __name__ == '__main__':
    creds = import_credentials()
    df_gulong = get_gulong_data()
    
    url = st.text_input('Enter Google Sheet URL')
    
    confirm_btn = st.button('Confirm')
    
    if len(url) and confirm_btn:
    
        df = import_sheet(url, creds)
        
        df_data = clean_data(df, df_gulong)
        
        # df_temp = pd.concat([df, df_data], axis=1)
        # df_temp2 = pd.concat([df_temp.iloc[:, :25], df_temp.loc[:, ['correct_specs', 'similar_pattern']]], axis=1)
        df_merged = match_df(df_data, df_gulong)
        
        st.dataframe(df_merged)
        
        if len(df_merged):
            st.download_button('Download table',
                               data = convert_df(df_merged),
                               file_name = 'gulong_matches.csv',
                               key = 'gulong_match')
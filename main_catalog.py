# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Carlo Solibet
"""

import pandas as pd
import numpy as np
from fuzzywuzzy import process, fuzz

import cleaner_functions as clean_func

def get_gulong_data() -> pd.DataFrame:
    '''
    Get gulong.ph data from backend
    
    Returns
    -------
    df : dataframe
        Gulong.ph product info dataframe
    '''
    show_cols = ['sku_name', 'raw_specs', 'price_gulong', 'name', 'brand', 
                 'width', 'aspect_ratio', 'diameter', 'correct_specs']
    
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
                                'promo' : 'price_gulong'}).reset_index(drop = True)
        
        ## 3. Perform data filtering and cleaning
        df.loc[df['sale_tag']==0, 'price_gulong'] = df.loc[df['sale_tag']==0, 'srp']
        # 1st pass
        df = df[df.width.notna() & df.aspect_ratio.notna() & df.diameter.notna() &\
                df.product_id.notna() & (df.brand.notna() & (df.brand != ' ')) &\
                    (df.activity == 1)]
        
        df.loc[:, 'width'] = df.apply(lambda x: clean_func.clean_width(x['width']), axis=1)
        df.loc[:, 'aspect_ratio'] = df.apply(lambda x: clean_func.clean_aspect_ratio(x['aspect_ratio']), axis=1)    
        df.loc[:, 'diameter'] = df.apply(lambda x: clean_func.clean_diameter(x['diameter']), axis=1)
        # 2nd pass
        df = df[df.width.notna() & df.aspect_ratio.notna() & df.diameter.notna() &\
                df.product_id.notna() & (df.brand.notna() & (df.brand != ' ')) &\
                    (df.activity == 1)]
        
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
        
        match = process.extractOne(s, ref.name, scorer = scorer, 
                           score_cutoff = cutoff)
        if match:
            result = match[0]
            try:
                brand = ref.loc[match[-1], 'brand']
            except:
                brand = np.NaN
        else:
            result = np.NaN
            brand = np.NaN
    except:
        result = np.NaN
        brand = np.NaN
    
    if with_brand:
        return pd.Series([result, brand])
    else:
        return result

def supplier_clean(ws : pd.DataFrame, 
                   supplier : str,
                   ref : pd.DataFrame) -> pd.DataFrame:
    '''
    Custom dataframe cleaning procedure for each supplier format
    
    Parameters:
    -----------
        - ws : pd.DataFrame
            supplier dataframe derived from worksheet
        - supplier : str
            supplier name
    
    Returns:
    --------
        - df_list : list
            list of dataframes as rows for contatenation
    
    '''
    if supplier == 'DRAKESTER INCORPORATED':
        # get index of rows with column names using 'SIZE'
        cols_ndx = []
        for r in ws.iterrows():
            if 'SIZE' in r[1].values: # r[1] is the series, r[0] is index
                cols_ndx.append(r[0])
        
        cols_ndx.append(len(ws))
        
        # partition dataframes
        df_list = []
        for ndx, c in enumerate(cols_ndx[:-1]):
            temp = ws.loc[cols_ndx[ndx]:cols_ndx[ndx+1]-1,:]
            ## remove NaN (whole rows and columns with NaN)
            temp = temp.dropna(axis = 0, how = 'all').dropna(axis = 1, how = 'all')
            # set first row as columns
            temp.columns = ['_'.join(v.lower().split(' ')) for v in temp.iloc[0].values]
            # set next row as first row
            temp = temp.iloc[1:, :]
            # rename columns
            temp = temp.rename(columns = {'type' : 'pattern',
                                      'net_price' : f'price_{supplier.upper()}',
                                      'srp' : f'price_{supplier.upper()}',
                                      'quantity' : f'qty_{supplier.upper()}'})
            # append to list
            df_list.append(temp.iloc[:,-5:])
        
        # concatenate partition dfs
        df = pd.concat(df_list)
        
        # remove duplicates
        df = df.drop_duplicates(keep = 'first').reset_index(drop = True)

        # standardize data
        for c in df.columns:
            # if data is numeric, do nothing
            # if data is alphabet, apply upper
            try:
                df.loc[:, c] = df.loc[:, c].apply(lambda x: float(x) if x.isnumeric() else x.upper())
            except:
                pass
        
        # 3. standardize sizes
        # use extract dimensions from cleaner functions
        df['correct_specs'] = df['size'].apply(lambda x: clean_func.combine_specs(*clean_func.clean_tire_size(x), mode = 'MATCH'))

        # 4. determine matching similar model names from reference
        df[['similar_pattern', 'brand']] = df.apply(lambda x: name_match(x['pattern'], ref),
                                                                 axis = 1)
    
    elif supplier == 'ABANTE TIRE MARKETING CORPORATION':
        
        cols_ndx = []
        for r in ws.iterrows():
            if 'Item Name' in r[1].values: # r[1] is the series, r[0] is index
                cols_ndx.append(r[0])
                break
        
        if cols_ndx:
            df = ws.loc[cols_ndx[0]+1:,:]
        else:
            df = ws
        
        ## remove NaN (whole rows and columns with NaN)
        df = df.dropna(axis = 0, how = 'all').dropna(axis = 1, how = 'all')
        # rename columns
        df.columns = ['sku_name', f'qty_{supplier}']
        # extract correct specs
        df['correct_specs'] = df['sku_name'].apply(lambda x: clean_func.combine_specs(*clean_func.clean_tire_size(x), mode = 'MATCH'))
        # extract brand
        df['brand'] = df['sku_name'].apply(lambda x: clean_func.clean_make(x, ref.brand.unique()))
        # extract pattern
        df['pattern'] = df['sku_name'].apply(lambda x: clean_func.clean_model(x, ref))
        # get match similar_pattern
        df['similar_pattern'] = df['sku_name'].apply(lambda x: name_match(x, ref, with_brand = False))
        
    return df
        

def extract_supplier_data(file : str or pd.DataFrame, 
                          df_gulong : pd.DataFrame = None, 
                          supplier : str = 'DRAKESTER INCORPORATED'):
    
    # 1. load supplier file
    try:
        wb = pd.ExcelFile(file)
        ws = pd.read_excel(wb, wb.sheet_names[0]).reset_index(drop = True)
    except:
        if isinstance(file, pd.DataFrame):
            ws = file
        else:
            ws = pd.read_csv(file)
    
    if df_gulong is None:
        df_gulong = get_gulong_data()
    
    # 2. clean worksheet and extract data
    ## clean whitespace
    ws = ws.replace('^(\s)*$', np.NaN, regex = True)
    
    # 3. clean worksheet dataframe
    df = supplier_clean(ws, supplier = supplier,
                        ref = df_gulong)
    
    # standardize data
    for c in df.columns:
        # if data is numeric, do nothing
        # if data is alphabet, apply upper
        try:
            df.loc[:, c] = df.loc[:, c].apply(lambda x: float(x) if x.isnumeric() or pd.isna(x) else x.upper())
        except:
            pass
    
    # remove duplicates
    df = df.drop_duplicates(keep = 'first').reset_index(drop = True)
    
    # 5. set supplier
    df['supplier'] = supplier
    
    return df
    

def get_supplier_data_from_dict(files : dict or list,
                                supp : str = None,
                                df_gulong : pd.DataFrame = None) -> pd.DataFrame:
    '''
    
    Looper function to iterate extraction and cleaning of supplier data
    per file per supplier
    
    Parameters:
    -----------
        - files : dict or list
            Input object of files in dict or list
        - supp : str
            Supplier name; needed only if files is type list
    
    Returns:
    --------
        - df_supplier : pd.DataFrame or None
    
    '''
    supplier_df_list = []
    
    # input is dictionary with keys as supplier, values as list of files
    if isinstance(files, dict):
        for supp in files.keys():
            for file_path in files[supp]:
                #print(f'Processing {supp}-{file_path}..')
                df_temp = extract_supplier_data(file_path, 
                                                df_gulong = df_gulong,
                                                supplier = supp)
                supplier_df_list.append(df_temp)
    # input is a list of files   
    elif isinstance(files, list):
        for file_path in files[supp]:
            #print(f'Processing {supp}-{file_path}..')
            df_temp = extract_supplier_data(file_path,
                                            df_gulong = df_gulong,
                                            supplier = supp)
            supplier_df_list.append(df_temp)
    
    else:
        return None
    
    # concatenate all supplier df
    df_supplier = pd.concat(supplier_df_list).reset_index(drop = True)

    return df_supplier

def match_df(df1, df2):
    '''
    Merge dataframes 
    
    '''
    
    # merge
    merged = df1.merge(df2, left_on = ['name', 'correct_specs', 'brand'],
                    right_on = ['similar_pattern', 'correct_specs', 'brand']).drop_duplicates()

    merged = merged.drop(columns = ['pattern', 'similar_pattern',
                                    'size', 'max'],
                         axis = 1)
    return merged

if __name__ == "__main__":
    df_gulong = get_gulong_data()
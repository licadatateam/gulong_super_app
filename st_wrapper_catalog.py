# -*- coding: utf-8 -*-
"""
Created on Sun May 12 12:10:14 2024

@author: carlo
"""

import streamlit as st
import pandas as pd
import main_catalog

def display_files():
    if len(st.session_state['files']):
        st.caption('To remove the file(s), select the corresponding checkbox.')
        
    for supplier in st.session_state['files']:
        st.write(f'All files uploaded under {supplier}:')
        for file in st.session_state['files'][supplier]:
            try:
                remove_file = st.checkbox(file.name,
                                          key = file.name)
            except:
                pass
            
            finally:
                if remove_file:
                    st.session_state['files'][supplier].remove(file)
                    st.rerun()


def upload_files(supp, num):
    supp_files = st.file_uploader(label = f'Upload {supp} files',
                          type = ['xlsx', 'csv', 'xls'],
                          accept_multiple_files = True,
                          key = supp + str(num))
  
    return supp_files

def files_summary():
    
    for supp in st.session_state['files'].keys():
        st.write(f'{supp} : Total = {len(st.session_state["files"][supp])}')
        for file in st.session_state['files'][supp]:
            st.write(f'--{file.name}')

@st.cache_data
def get_supplier_data(files : dict or list,
                      supp : str = None,
                      df_gulong : pd.DataFrame = None):
    
    df_supplier = main_catalog.get_supplier_data_from_dict(files)
    return df_supplier

@st.cache_data
def get_supplier_names():
    # get list of suppliers
    url1 =  "http://app.redash.licagroup.ph/api/queries/131/results.csv?api_key=FqpOO9ePYQhAXrtdqsXSt2ZahnUZ2XCh3ooFogzY"
    df_data = pd.read_csv(url1)
    
    suppliers = sorted(df_data[(df_data.name.notna()) &\
                               (df_data.activity == 1)].name.unique())
    
    return suppliers

def main():
    
    suppliers = get_supplier_names()
    
    if 'files' not in st.session_state:
        st.session_state['files'] = {}
    
    # upload supplier files
    supplier_select = st.selectbox('Select Supplier Files',
                     options = suppliers,
                     index = list(suppliers).index('DRAKESTER INCORPORATED'))

    if (supplier_select not in st.session_state['files'].keys()) or len(st.session_state['files'][supplier_select]) == 0:
        supp_files = upload_files(supplier_select, 0)
        if len(supp_files):
            st.session_state['files'][supplier_select] = supp_files
            st.rerun()
    
    else:
        supp_files = upload_files(supplier_select, 
                                  len(st.session_state['files'][supplier_select]))
        if len(supp_files):
            st.session_state['files'][supplier_select].extend(supp_files)
            st.rerun()
            
    # review uploaded files
    #with st.expander('Uploaded Files Summary'):
    display_files()
    
    if st.button('Confirm'):
        
        if len(st.session_state['files']):
        
            try:
                df_supplier = get_supplier_data(st.session_state['files'])
                #df_supplier = main_catalog.get_supplier_data_from_dict(st.session_state['files'])
                
                # df_gulong = main_catalog.get_gulong_data()
                # merged = main_catalog.match_df(df_gulong, df_supplier)
                # st.write(merged)
                
                return df_supplier
            
            except Exception as e:
                st.exception(e)
                return None
        
        else:
            st.error('No supplier files were uploaded.')
            return None
    
    

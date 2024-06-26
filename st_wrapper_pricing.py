# -*- coding: utf-8 -*-
"""
Created on Wed May  8 22:10:35 2024

@author: carlo
"""
import os, sys
import json
import numpy as np
import pandas as pd
from datetime import datetime as dt
from io import BytesIO


import main_pricing_2 as mp
import st_wrapper_catalog
import gulong_sku_match

import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode, DataReturnMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import extra_streamlit_components as stx

st.set_page_config(layout="wide")

# load google service account
try:
    creds = st.secrets['secrets']
except:
    with open('secrets.json') as file:
        creds = json.load(file)

# configure working directory
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
output_path = os.path.abspath(os.path.dirname(__file__))
os.chdir(output_path)

@st.cache_data
def to_float(num : (int, float, str)):
    '''
    Check if value is convertible to float
    
    Parameters:
    -----------
        - num : int, str, float
    
    Returns:
    -------
        - bool : True or False
    '''
    try:
        float(num)
        return True
    except ValueError:
        return False


@st.cache_data
def acquire_data():
    '''
    Wrapping function of acquire_data to apply caching
    '''
    return mp.acquire_data()

@st.cache_data
def implement_sale(df : pd.DataFrame, 
                   sale_tag : str, 
                   promo : str, 
                   srp : str) -> pd.DataFrame:
    '''
    Applies SRP to non-sale items
    
    Parameters:
    -----------
        df : pd.DataFrame
            dataframe to modify (df_final from acquire_data())
        sale_tag : str
            column name of sale tag
        promo : str
            column name of Gulong PH promo price (GulongPH)
        srp : str
            column name of Gulong PH srp (GulongPH_slashed)
    
    Returns:
    --------
        - df : pd.DataFrame
    
    '''
    df.loc[df[sale_tag]==0, promo] = df.loc[df[sale_tag]==0, srp]
    return df

def update():
    '''
    Resets cache data and runtime
    '''
    st.cache_data.clear()
    del st.session_state['adjusted']
    del st.session_state['GP_15']
    del st.session_state['GP_20a']
    del st.session_state['GP_20b']
    st.experimental_rerun()

def set_session_state(updated_at : str = None):
    '''
    Initializes session state
    
    Parameters:
    -----------
        None
    Returns:
    --------
        None
    
    '''
    
    if 'updated_at' not in st.session_state:
        if updated_at is None:
            st.session_state['updated_at'] = dt.today().date().strftime('%Y-%m-%d')
        else:
            st.session_state['updated_at'] = updated_at

    if 'GP_15' not in st.session_state:
        st.session_state['GP_15'] = 15
    if 'GP_20' not in st.session_state:
        st.session_state['GP_20a'] = 5
    if 'GP_20_' not in st.session_state:
        st.session_state['GP_20b']= 3
    if 'd_b2b' not in st.session_state:
        st.session_state['d_b2b'] = 25
    if 'd_affiliate' not in st.session_state:
        st.session_state['d_affiliate'] = 27
    if 'd_marketplace' not in st.session_state:
        st.session_state['d_marketplace'] = 25
    
    if 'reload_data' not in st.session_state:
        st.session_state['reload_data'] = False
    
    if 'adjusted' not in st.session_state:
        st.session_state['adjusted'] = False
        
    if 'updated_at' not in st.session_state:
        update()

def quick_calculator():
    '''
    Sidebar tool for quick calculations around selling price, supplier price
    and GP
    
    Parameters:
    ----------
        None.
    
    Returns:
    --------
        None.
    
    '''
    find_value = st.radio("Find:", ('Selling Price', 
                                    'Supplier Price', 
                                    'GP(%)'))
    q1,q2 = st.columns([1,1])
    if find_value =='Selling Price':     
        with q1:
            qsp = st.text_input('Supplier Price:', 
                                value="1000.00")
        with q2:
            qgp = st.text_input('GP: (%)', 
                                value="30.00")
            if to_float(qgp) and to_float(qsp):
                value = mp.consider_GP(float(qsp),
                                    float(qgp))
            else:
                value = "Input Error"
                
    if find_value =='Supplier Price':       
        with q1:
            qsp = st.text_input('Selling Price:', 
                                value="1000.00")
        with q2:
            qgp = st.text_input('GP (%):', 
                                value="30.00")
            
            if to_float(qgp) and to_float(qsp):
                value = round(float(qsp)/(1+float(qgp)/100),)
            else:
                value = "Input Error"
                
    if find_value == 'GP(%)':
        with q1:
            qsp = st.text_input('Selling Price:', 
                                value="1500.00")
        with q2:
            qsupp = st.text_input('Supplier Price:', 
                                  value="1000.00")
            
            if (to_float(qsupp) and to_float(qsp)):
                if float(qsp)==0:
                    value = "Input Error"
                else:
                    value = mp.get_GP(qsupp, qsp)
            else:
                value = "Input Error"
    # show
    st.metric(find_value, value)
    
def rename_tiers() -> dict:
    '''
    Sidebar option to rename tiers
    
    Parameters:
    -----------
        None
    
    Returns:
    --------
        - tier_names : dict
    
    '''
    
    t1_name = st.text_input('Tier 1 name:', 'Website Slashed Price Test')
    t2_name = st.text_input('Tier 2 name:', 'Website Prices Test')
    t3_name = st.text_input('Tier 3 name:', 'B2B Test')
    t4_name = st.text_input('Tier 4 name:', 'Marketplace Test')
    t5_name = st.text_input('Tier 5 name:', 'Affiliates Test')
    tier_names = {'tier1' : t1_name,
             'tier2' : t2_name,
             'tier3' : t3_name,
             'tier4' : t4_name,
             'tier5' : t5_name}
    return tier_names

@st.cache_data
def adjust_wrt_gogulong(df : pd.DataFrame,
                        GP_15 : int = 15,
                        GP_20a : int = 5,
                        GP_20b : int = 1, 
                        b2b : int = 25,
                        affiliate : int = 27,
                        marketplace : int = 25) -> pd.DataFrame:
    '''
    Wrapping function of adjust_wrt_gogulong to apply caching
    '''
    return mp.adjust_wrt_gogulong(df,
                            GP_15,
                            GP_20a,
                            GP_20b,
                            b2b,
                            affiliate,
                            marketplace)


def preorder_calc(qty_list : list) -> bool or float:
    '''
    Determines whether to preorder or not based on available
    quantity/stocks data
    
    Parameters:
    -----------
        - qty_list : list
            list of float or int values
    
    Returns:
    --------
        - bool or float : True/False or np.nan if quantity data is all NaN
    
    '''
    nan_count = np.sum([1 for n in qty_list if pd.isna(n)])
    if nan_count == len(qty_list):
        return np.nan
    else:
        if np.nansum(qty_list) <= 4:
            return 'True'
        else:
            return 'False'

def filter_data_captured(df_test : pd.DataFrame, 
                         tier : list) -> pd.DataFrame:
    df_compet = pd.DataFrame()
    if 'GoGulong' in df_test.columns:
        df_temp = df_test.loc[df_test['GulongPH']>df_test['GoGulong']]
        df_compet = pd.concat([df_compet,df_temp])
        
    if 'TireManila' in df_test.columns:
        df_temp = df_test.loc[df_test['GulongPH']>df_test['TireManila']]
        df_compet = pd.concat([df_compet,df_temp])
        
    df_A = df_test.loc[df_test['supplier_max_price']> df_test[['GulongPH','b2b','marketplace']].min(axis=1)]
    df_B = df_test.loc[df_test['GulongPH']> df_test[['GulongPH_slashed','marketplace']].min(axis=1)]
    df_C = pd.DataFrame()
    
    for col in tier:
        df_E = df_test.loc[df_test['supplier_max_price']> df_test[col]]
        df_C = pd.concat([df_C,df_E],axis=0)
    df_show = pd.concat([df_A,df_compet,df_B,df_C],axis=0)
    df_show = df_show.drop_duplicates()
    
    return df_show

def build_grid(df_show : pd.DataFrame):
    '''
    Configures GridOptionsBuilder
    
    Parameters:
    -----------
        - df_show : pd.DataFrame
    
    Returns:
        - response : AgGrid object
    
    '''
    gb = GridOptionsBuilder.from_dataframe(df_show)
    gb.configure_columns(autoSizeAllColumns = True,
                         filterable = True)
    gb.configure_default_column(enablePivot=False, 
                                enableValue=False, 
                                enableRowGroup=False, 
                                editable = True)
    def_col = 'model' if 'sku_name' not in df_show.columns else 'sku_name'
    gb.configure_column(def_col, headerCheckboxSelection = True,
                        pinned = True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()  
    gridOptions = gb.build()
    
    # show table
    response = AgGrid(df_show,
        #theme = 'light',
        gridOptions=gridOptions,
        height = 500,
        #width = '100%',
        editable=True,
        allow_unsafe_jscode=True,
        reload_data = st.session_state['reload_data'],
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False)
    
    # reset
    st.session_state['reload_data'] = False
    
    return response

def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')


def highlight_promo(xa : pd.DataFrame):
    df1 = pd.DataFrame('background-color: ', index = xa.index, 
                       columns = xa.columns)
    col_eval = ['GulongPH','GulongPH_slashed','b2b','marketplace']
    highlight_competitor = '#ffffb3'
    temp_list = list(col_tier)
    col_eval = col_eval+temp_list
    for column in col_eval:
        c = xa['supplier_max_price'] > xa[column]
        df1['supplier_max_price']= np.where(c, 'background-color: {}'.format('pink'), df1['supplier_max_price'])
        df1[column]= np.where(c, 'background-color: {}'.format('pink'), df1[column])
    if 'selection_max_price' in xa.columns.tolist():
        c = xa['selection_max_price']<xa['supplier_max_price']
        df1['selection_max_price'] = np.where(c, 'background-color: {}'.format('lightgreen'), df1['selection_max_price'])
    if 'GoGulong' in xa.columns.tolist():
        
        c = xa['GulongPH']>xa['GoGulong']
        df1['GulongPH'] = np.where(c, 'background-color: {}'.format(highlight_competitor), df1['GulongPH'])
        df1['GoGulong'] = np.where(c, 'background-color: {}'.format(highlight_competitor), df1['GoGulong'])
    if 'TireManila' in xa.columns.tolist():
        c = xa['GulongPH']>xa['TireManila']
        df1['TireManila'] = np.where(c, 'background-color: {}'.format(highlight_competitor), df1['TireManila'])
        df1['GulongPH'] = np.where(c, 'background-color: {}'.format(highlight_competitor), df1['GulongPH'])
    return df1

def highlight_others(x):#cols = ['GP','Tier 1','Tier 3', etc]
    df1 = pd.DataFrame('background-color: ', index=x.index, columns=x.columns)
    c1 = x['GulongPH'] > x['GulongPH_slashed']
    df1['GulongPH']= np.where(c1, 'color:{};font-weight:{}'.format('red','bold'), df1['GulongPH'])
    df1['GulongPH_slashed']= np.where(c1, 'color:{};font-weight:{}'.format('red','bold'), df1['GulongPH_slashed'])
    c2 = x['marketplace']<x['GulongPH']
    df1['GulongPH']= np.where(c2, 'color:{};font-weight:{}'.format('red','bold'), df1['GulongPH'])
    df1['marketplace']= np.where(c2, 'color:{};font-weight:{}'.format('red','bold'), df1['marketplace'])
    return df1

def highlight_smallercompetitor(xa):
    df1 = pd.DataFrame('background-color: ', index=xa.index, columns=xa.columns)
    col_eval = ['GoGulong','TireManila','PartsPro']
    for column in col_eval:
        if column in xa.columns:
            c = xa['GulongPH'] > xa[column] # filter condition
            df1['GulongPH']= np.where(c, 'background-color: {}'.format('pink'), df1['GulongPH'])
            df1[column]= np.where(c, 'background-color: {}'.format('pink'), df1[column])
    return df1

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.close()
    processed_data = output.getvalue()
    return processed_data

if __name__ == "__main__":
    
    ## Sidebar Tool : Quick Calculator
    qc_expander = st.sidebar.expander("Quick Calculator", expanded = False)
    with qc_expander:
        quick_calculator()
    
    # Sidebar Tool : Rename Tiers
    t_name= st.sidebar.expander("Rename Tiers:", expanded = False)
    with t_name:
        tier_names = rename_tiers()
        
    # Load data
    # keys : df_final, cols_option, df_competitor, last_update
    data_dict = acquire_data()
    # initialize session state
    set_session_state(updated_at = data_dict['backend_last_update'])
    
    chosen_tab = stx.tab_bar(data = [
        stx.TabBarItemData(id = '1', title = 'Pricing', description = ''),
        stx.TabBarItemData(id = '2', title = 'Stock Level Check', description = ''),
        stx.TabBarItemData(id = '3', title = 'SKU Matching', description = ''),
        ], default = '1')
    
    placeholder = st.container()
    
    # pricing
    if chosen_tab == '1':
        with placeholder:
        
            # displays update status of data displayed
            upd_btn, upd_data = st.sidebar.columns([2,3])
            with upd_btn:
                help_note = 'Acquires results of query from backend and resets program'
                if st.button('Update Data', help = help_note):
                    update()
            with upd_data:
                st.caption(f'Gulong PH data last updated on {data_dict["backend_last_update"]}')
                comp_update = str(data_dict['comp_last_update']).replace('/', '-')
                st.caption(f'Competitor data last updated on {comp_update}.')
            
            st.sidebar.markdown("""---""")
            
            # auto-adjust values toggle
            auto_adj_btn, auto_adj_txt = st.sidebar.columns([2,3])
            with auto_adj_btn:
                is_adjusted = st.checkbox('Auto-adjust')
            with auto_adj_txt:
                st.caption('Automatically adjusts data based on GoGulong values')
            if is_adjusted:
                st.session_state['adjusted'] = True
            else:
                st.session_state['adjusted'] = False
            
            if st.session_state['adjusted']:
                # adjust srp price to non-sale gulong items
                # set to only active items
                df_final_ = implement_sale(data_dict['df_final'][data_dict['df_final'].activity == 1], 
                                           'sale_tag', 
                                           'GulongPH', 
                                           'GulongPH_slashed').drop(columns= 'sale_tag')
                
                df_final_ = df_final_.set_index('model')
                # adjust gulong prices with respect to gogulong
                df_temp_adjust = adjust_wrt_gogulong(df_final_,
                                                     st.session_state['GP_15'],
                                                     st.session_state['GP_20a'],
                                                     st.session_state['GP_20b'],
                                                     st.session_state['d_b2b'],
                                                     st.session_state['d_affiliate'],
                                                     st.session_state['d_marketplace'])
                # update values
                df_final_.update(df_temp_adjust[['GulongPH',
                                                'GulongPH_slashed',
                                                'b2b',
                                                'affiliate',
                                                'marketplace']], 
                                overwrite = True)
                df_final = df_final_.reset_index()
            else:
                df_final = data_dict['df_final'].copy()

            # edit mode
            edit_mode = st.sidebar.selectbox('Mode', 
                                             options = ('Automated',
                                                        'Manual'),
                                             index = 1)
            
            check_adjusted = st.sidebar.checkbox('Show adjusted prices only', 
                                                 value = False)
            
            # Manual edit mode
            if edit_mode == 'Manual':
                st.header("Data Review")
                
                with st.expander('Include/remove columns in list:'):
                    beta_multiselect = st.container()
                    check_all = st.checkbox('Select all', value=False)
                    # list of default columns to show
                    def_list = list(data_dict['cols_option']) if check_all else []
                        
                    selected_cols = beta_multiselect.multiselect('Included columns in table:',
                                                   options = data_dict['cols_option'],
                                                   default = def_list)
                    selected_cols = list(set(selected_cols))
                
                
                df_show = df_final.merge(data_dict['df_final'][['model', 'GulongPH']], 
                                               how = 'left',
                                               on = 'model', 
                                               suffixes=('', '_backend'))
                
                if check_adjusted:
                    df_show = df_show.loc[df_show['GulongPH']
                                          != df_show['GulongPH_backend']]
                
                # default table columns
                cols = ['model_','model','make', 'pattern', 'dimensions', 'supplier_max_price',
                        '3+1_promo_per_tire_GP25','GulongPH','GulongPH_slashed',
                        'b2b','marketplace', 'GoGulong', 'TireManila', 'PartsPro',
                        'qty_tiremanila', 'year']
                
                # TODO: reorder columns (qty and price)
                
                if len(selected_cols) > 0:
                    # add selected cols to display
                    cols.extend(selected_cols)
                
                # final dataframe to show
                df_show = df_show[df_show.activity == 1].dropna(how = 'all', 
                                                                subset = cols,
                                                                axis=0).replace(np.nan,'')
                
                df_show = df_show[cols].drop(columns='model').rename(
                                                            columns={'model_': 'sku_name'})
                
                st.write("""Select the SKUs that would be considered for the computations.
                         Feel free to filter the _make_ and _model_ that would be shown. 
                         You may also select/deselect columns.""")
                         
                reset_btn, reset_capt = st.sidebar.columns([1, 1])
                with reset_btn:
                    if st.button('Reset changes'):
                        st.session_state['reload_data'] = True
                with reset_capt:
                    st.caption('Resets the edits done in the table.')
                
                
                # build and show table
                response = build_grid(df_show)
                
                AG1, AG2 = st.columns([3,2])
                with AG1:
                    st.write(f"Results: {len(df_show)} entries")
                with AG2:
                    st.download_button(label="📥 Download this table.",
                                        data=convert_df(pd.DataFrame.from_dict(response['data'])),
                                        file_name='grid_table.csv',
                                        mime='text/csv')
                
                st.markdown("""
                            ---
                            """)
                            
                st.header("Price Comparison")   
                st.write("""You may set the GP and the price comparison between models 
                         would be shown in a table.""")
                         
                ct1, ct2,ct3,ct4,ct5, cs3= st.columns([1,1,1,1,1,1])#,cS,cs1,cs2,cs3
                with ct1:
                    test_t1 = st.checkbox(tier_names['tier1'])
                    t1_GP = st.text_input("GP (%):", value = "30", key='t1')
                with ct2:
                    test_t2 = st.checkbox(tier_names['tier2'])
                    t2_GP = st.text_input("GP (%):", value = "27", key='t2')
                with ct3:
                    test_t3 = st.checkbox(tier_names['tier3'])
                    t3_GP = st.text_input("GP (%):", value = "25", key='t3')
                with ct4:
                    test_t4 = st.checkbox(tier_names['tier4'])
                    t4_GP = st.text_input("GP (%):", value = "28", key='t4')
                with ct5:
                    test_t5 = st.checkbox(tier_names['tier5'])
                    t5_GP = st.text_input("GP (%):", value = "27", key='t5')
                
                with cs3:
                    GP_promo = st.text_input("3+1 Promo GP (%):", 
                                             value= "25", key='t_promo')
                
                # work with checked rows in table
                df = pd.DataFrame.from_dict(response['selected_rows'])
                # mask for checked boxes
                col_mask    = [test_t1, test_t2, test_t3, test_t4, test_t5]
                # column names checked
                col_tier    = list(np.array(list(tier_names.values()))[col_mask])
                # input GP selected
                col_GP      = list(np.array([t1_GP, t2_GP, t3_GP, 
                                             t4_GP, t5_GP])[col_mask])
                
                temp_list = [to_float(input_GP) for input_GP in col_GP]
                
                captured_vals_only = st.sidebar.checkbox("""Show captured erroneous 
                                                         values only.""")
                st.sidebar.caption("""Program may run slow when unchecked. Uncheck 
                                   before saving for website template.""")    
                st.sidebar.markdown("""---""")
                
                temp_list_ = []
                for input_GP in col_GP:
                    temp_list_.append(to_float(input_GP))
                
                # check
                if len(temp_list_) != sum(temp_list_):
                    st.error("Input Error")
                    st.stop()
                    
                else:
                    if len(df) > 0:
                        # clean
                        df = df.drop(['make', 'dimensions', 
                                      'pattern'], axis = 1).set_index('sku_name')
                        df = df.replace('', np.nan).dropna(axis=1, how='all')
                        for c in ["rowIndex", "_selectedRowNodeInfo"]:
                            try:
                                df = df.drop(c, axis = 1)
                            except:
                                pass
                        # convert to numeric
                        df = df.apply(pd.to_numeric, errors = 'ignore')
            
                        drop_cols = ['supplier_max_price', 
                                     'GulongPH', 'GulongPH_slashed', 'b2b','marketplace',
                                     'GoGulong', 'GoGulong_slashed', 'TireManila',
                                     'qty_tiremanila', 'year']
                         
                        # isolate columns to eval
                        column_eval = list(set(df.columns) - set(drop_cols))
                        
                        # if column_eval is non empty and reasonable amount of selected cols
                        if len(column_eval) > 0 and \
                                (len(selected_cols) < len(data_dict['cols_option'])):
                            # get max of checked columns
                            df['selection_max_price'] = df[column_eval].fillna(0)\
                                                        .apply(lambda x: x.max(), axis=1)
                        
                        # apply GP for each selected model
                        for c in range(len(col_tier)):
                            df[col_tier[c]] = df['supplier_max_price']\
                                            .apply(lambda x: mp.consider_GP(x, col_GP[c]))
                        
                        if captured_vals_only: 
                            df = filter_data_captured(df, col_tier)
                        
                        df['3+1_promo_per_tire'] = df['supplier_max_price'].apply(lambda x: mp.promotize(x,GP_promo))
                        # show
                        st.dataframe(df.style.apply(highlight_promo, axis=None)\
                                     .apply(highlight_others,axis=None)\
                                     .apply(highlight_smallercompetitor,axis=None)\
                                     .format(precision = 2))
                    
                    else:
                        st.info("Kindly check/select at least one above.")
            
                CPC1, CPC2 = st.columns([3,2])
                with CPC1:
                    st.write(f'Showing {len(df)} out of {len(df_show)} entries.')
                with CPC2:
                    if len(df)>0:
                        to_csv = convert_df(df)
                        st.download_button(label="📥 Download this table as csv",
                                            data=to_csv,
                                            file_name='price_comparison.csv',
                                            mime='text/csv')
            
            if edit_mode == 'Automated':
                st.header("Automation parameters")
                with st.expander('Show comparative data:'):
                    st.info('Number of SKUs where competitors are cheaper than GulongPH:')
                    # Set names of competitors
                    base_cols = ['make', 'model', 'GulongPH']
                    comps = ['GoGulong', 'TireManila', 'PartsPro']
                    
                    df_showsummary = df_final[base_cols + comps]
                    
                    # create columns for each competitor
                    comp_cols = st.columns([1]*len(comps))
                    for ndx, comp in enumerate(comp_cols):
                        with comp_cols[ndx]:
                            # Count SKUs which are cheaper than Gulong PH
                            comp_cheap = len(df_showsummary.loc[df_showsummary['GulongPH']>\
                                                                df_showsummary[comps[ndx]]])
                            # show competitor and number of SKUs
                            st.metric(comps[ndx], 
                                      comp_cheap)
                    
                    # dict for file download parameters
                    csv_dict = {'adjusted' : {'df' : data_dict['df_final'][base_cols + comps],
                                              'filename' : 'adjusted_price_comparison.csv',
                                              'label' : "📥 Download table below as csv",
                                              }, 
                                'raw' : {'df' : df_final[base_cols + comps],
                                         'filename' : 'raw_price_comparison.csv',
                                         'label' : "📥 Download raw comparison table as csv"}
                                }
                    
                    # prep UI for download files
                    file_cols = st.columns([1, 1])
                    for ndx, f in enumerate(list(csv_dict.keys())):
                        with file_cols[ndx]:
                            st.download_button(label = csv_dict[f]['label'],
                                            data = convert_df(csv_dict['adjusted']['df']),
                                            file_name = csv_dict[f]['filename'],
                                            mime = 'text/csv')
                    # show table
                    st.dataframe(df_showsummary.style.apply(highlight_smallercompetitor, 
                                                        axis=None)\
                                                 .format(precision = 2),
                                                 use_container_width = True)
                
                gp_dict = {'b2b' : [],
                           'affiliate' : [],
                           'marketplace' : []}
                           
                # set GP for platforms
                gp_cols = st.columns([1,1,1,1])
                for ndx, col in enumerate(list(gp_dict.keys())):
                    with gp_cols[ndx]:
                        gp_dict[col] = st.text_input(f'Set {col.upper()} GP:',
                                                     value = st.session_state[f'd_{col}'])
                        
                        if to_float(gp_dict[col]):
                            st.session_state[f'd_{col}'] = float(gp_dict[col])
                        else:
                            gp_dict[col] = st.session_state[f'd_{col}']
                            st.warning(f'Input error! {col.upper()} GP set to {gp_dict[col]}')
                
                with gp_cols[-1]:
                    apply_btn = st.button('Apply changes', key = 'apply1')
                
                with st.expander("Modify automated pricing rules with respect to GoGulong"):
                    
                    case_tabs = st.tabs(['Case 1', 'Case 2', 'Case 3'])
                    test_cols = ['model','supplier_max_price',
                                 'GulongPH','GoGulong','GulongPH_GP',
                                 'GoGulong_GP']
                    
                    with case_tabs[0]:
                        st.markdown("""
                                    #### If GoGulong GP < 15%, then set GulongPH GP to 15%.
                                    """)
                        
                        GP_15_raw = st.text_input('Set GP:', 
                                                  value = st.session_state['GP_15'], 
                                                  help ='Set GulongPH GP to this amount (%)')
                        
                        if to_float(GP_15_raw):
                            st.session_state['GP_15'] = float(GP_15_raw)
                            df_test1 = adjust_wrt_gogulong(df_final, 
                                                           GP_15 = float(GP_15_raw))
                            df_test1 = df_test1.loc[df_test1['GoGulong_GP'] < 15]
                            if len(df_test1) > 0:
                                e1a, e1b = st.columns([1,5])
                                with e1a:
                                    show_15 = st.checkbox('Show all', 
                                                          key = 'gp15')
                                    
            
                                
                                if show_15:
                                    df_show_15 = df_test1[test_cols].set_index('model')
                                    text = 'Showing all:'
                                
                                else:
                                    text = 'Example:'
                                    df_show_15 = df_test1[test_cols].head()\
                                                                    .set_index('model')
                                
                                st.dataframe(df_show_15.style.format(precision = 2))
                                st.caption(f'Showing {len(df_show_15)} of {len(df_test1)} changes.')
                                
                        else:
                            GP_15 = 15.0
                            st.warning('Input error! GP set to 15%')
                
                    with case_tabs[1]:
                        st.markdown("""
                                    #### If GoGulong GP is between 15% and 20%, match 
                                    GulongPH price. """)
                        
                        GP_20a_raw = st.text_input('Price offset value: ', 
                                                  value = st.session_state['GP_20a'], 
                                                  help='Decrease GoGulong by this amount (Php)')
                        if to_float(GP_20a_raw):
                            st.session_state['GP_20a'] = float(GP_20a_raw)
                            df_test2 = adjust_wrt_gogulong(df_final, 
                                                           GP_20a = st.session_state['GP_20a'])
                            df_test2 = df_test2.loc[(df_test2['GoGulong_GP']<20) &\
                                                    (df_test2['GoGulong_GP']>=15)]
                            if len(df_test2) > 0:
                                e2a, e2b = st.columns([1,5])
                                with e2a:
                                    show_20 = st.checkbox('Show all', key = 'gp20')
                                
                                if show_20:
                                    df_show_20 = df_test2[test_cols].set_index('model')
                                    text = 'Showing all:'
                                
                                else:
                                    text = 'Example:'
                                    df_show_20 = df_test2[test_cols].head().set_index('model')
                                
                                with e2b:
                                    st.write('Example:')
                                
                                st.dataframe(df_show_20.style.format(precision = 2))
                                st.caption('Showing '+str(len(df_show_20))+' of '+str(len(df_test2))+' changes.')
                        else:
                            GP_20a = 5.0
                            st.write(f'Input error! Price offset is {GP_20a}')
                            
                    with case_tabs[2]:
                        st.markdown("""
                                    #### If GoGulong GP > 20%, then adjust GP 
                                    correspondingly.
                                    """)
                        
                        GP_20b_raw = st.text_input('GP Offset value: ', 
                                                   value = st.session_state['GP_20b'], 
                                                   help = 'Decrease GoGulong GP by this amount (%)')
                        if to_float(GP_20b_raw):
                            st.session_state['GP_20b'] = float(GP_20b_raw)
                            df_test3 = adjust_wrt_gogulong(df_final, 
                                                           GP_20b = st.session_state['GP_20b'])
                            df_test3 = df_test3.loc[df_test3['GoGulong_GP']>=20]
                            if len(df_test3) >0:
                                e3a, e3b = st.columns([1,5])
                                with e3a:
                                    show_20_ = st.checkbox('Show all', key = 'gp20_')
                                
                                if show_20_:
                                    df_show_20_ = df_test3[test_cols].set_index('model')
                                    text = 'Showing all:'
                                
                                else:
                                    text = 'Example:'
                                    df_show_20_ = df_test3[test_cols].head().set_index('model')
                                with e3b:
                                    st.write('Example:')
                                st.dataframe(df_show_20_.style.format(precision = 2))
                                st.caption('Showing {len(df_show_20_)} of len(df_test3)) changes.')
                        else:
                            GP_20b = 1.0
                            st.write(f'Input error! GP offset is {GP_20b}')
                    st.button('Apply changes', key = 'apply2')
                
                if st.session_state['adjusted']: 
                    st.write("""Prices have been adjusted. The following Gulong.ph 
                             SKU prices are still greater than GoGulong's SKU prices.""")
                else:
                    st.write("""Kindly review the changes that would be implemented to 
                             the product prices.""")
                
                df_adjusted = adjust_wrt_gogulong(df_final,
                                                  st.session_state['GP_15'],
                                                  st.session_state['GP_20a'],
                                                  st.session_state['GP_20b'],
                                                  st.session_state['d_b2b'],
                                                  st.session_state['d_affiliate'],
                                                  st.session_state['d_marketplace'])
                
                if len(df_adjusted) > 0:
                    
                    df_temp_adj = df_final[['model', 'supplier_max_price', 
                                          'GulongPH']].merge(df_adjusted[['model',
                                                                          'GulongPH',
                                                                          'GoGulong',
                                                                          'GulongPH_GP',
                                                                          'GoGulong_GP',
                                                                          'GulongPH_slashed',
                                                                          'b2b',
                                                                          'affiliate',
                                                                          'marketplace']], 
                                                                          on='model',
                                                                          suffixes = ('_backend','_adjusted'), 
                                                                          how='right').set_index('model')
                    st.dataframe(df_temp_adj.style.format(precision = 2))#lsuffix = '_backend', rsuffix='_adjusted'
                    cRa, cRb = st.columns([2,1])
                    with cRa:
                        st.caption(f"""Reviewing {str(len(df_temp_adj))} changes to be 
                                   implemented out of {str(len(df_final.loc[df_final['GoGulong'].notnull()]))} 
                                   SKU overlaps.""")
                    
                    with cRb:
                        output = BytesIO()
                        writer = pd.ExcelWriter(output, engine=None)
                        df_temp_adj.reset_index().to_excel(writer, index=False, sheet_name='All changes')
                        df_test1[['model','supplier_max_price','GulongPH','GoGulong','GulongPH_GP','GoGulong_GP']].to_excel(writer, index=False, sheet_name='Case 1')
                        df_test2[['model','supplier_max_price','GulongPH','GoGulong','GulongPH_GP','GoGulong_GP']].to_excel(writer, index=False, sheet_name='Case 2')
                        df_test3[['model','supplier_max_price','GulongPH','GoGulong','GulongPH_GP','GoGulong_GP']].to_excel(writer, index=False, sheet_name='Case 3')
                        workbook = writer.book
                        writer.close()
                        processed_data = output.getvalue()
                        st.download_button(label='📥 Download this data',
                                                    data=processed_data ,
                                                    file_name= 'automated_changes.xlsx')
                else:
                    st.info('No changes to implement.')
                
            st.markdown("""
                        ---
                        """)
            
            dl_cols = st.columns(2)
            
            st.header('Download tables:')
            with dl_cols[0]:
                st.write("**Backend data:**")
                if df_final is not None:
                    csvA = convert_df(df_final)
                    st.download_button(
                                        label="backend_data.csv",
                                        data=csvA,
                                        file_name='backend_data.csv',
                                        mime='text/csv'
                                        )
            
            if edit_mode == 'Manual':
                with dl_cols[1]:
                    st.write("**All data:**")
                    if df_show is not None:
                        csvB = convert_df(df_show)
                        st.download_button(label="gulong_pricing.csv",
                                            data=csvB,
                                            file_name='gulong_pricing.csv',
                                            mime='text/csv')
            
            st.write("**Final data:**")        
            final_columns = ['Brand', 'TIRE_SKU', 
                    'Section_Width', 'Height', 'Rim_Size', 'Pattern',
                    'LOAD_RATING', 'SPEED_RATING', 'Supplier_Price', 'SRP', 'PROMO_PRICE',
                    'b2b_price', 'mp_price', 'STOCKS_ON HAND', 'Primary_Supplier']
             
            df_final = data_dict['df_final'].set_index('model')#.reset_index() [['make','model',
                    # 'section_width', 'aspect_ratio','rim_size','pattern',
                    # 'load_rating', 'speed_rating','supplier_max_price','GulongPH_slashed','GulongPH',
                    # 'b2b','marketplace''stock','supplier_id']]
            if edit_mode =='Manual' and len(df)>0:
                req_cols = ['sku_name','supplier_max_price','GulongPH_slashed','GulongPH','b2b','marketplace']
                req_cols.extend(col_tier)
                final_ = df.reset_index()[req_cols].set_index('sku_name')
                df_final.update(final_)

            final_df = data_dict['df_final'].reset_index()
            final_df['aspect_ratio'] = final_df['aspect_ratio'].fillna(0)
            final_df['speed_rating'] = final_df['speed_rating'].fillna(' ')       
            final_df = pd.concat([final_df.set_index('model'),df_final],axis=0)
            final_df = final_df[~final_df.index.duplicated(keep='first')].reset_index().sort_values(by='model')[['make','model',
                    'section_width', 'aspect_ratio','rim_size','pattern',
                    'load_rating', 'speed_rating','supplier_max_price','GulongPH_slashed','GulongPH',
                    'b2b','marketplace','stock','supplier_id']]
            
            df_temmmp_ = data_dict['df_final'][['model','supplier_max_price','GulongPH_slashed',
                                   'GulongPH','b2b','marketplace']].merge(df_final.reset_index()[['model','GulongPH_slashed','GulongPH','b2b','marketplace']], on='model',suffixes = ('_backend','_adjusted'), how='left')
            
            with st.expander('Review changes in prices'):
                r1,r2,r3,r4 = st.tabs(['SRP', 'Promo Price', 'B2B Price', 'Marketplace'])
                with r1:
                  st.dataframe(df_temmmp_.loc[df_temmmp_['GulongPH_slashed_backend']!=df_temmmp_['GulongPH_slashed_adjusted'],['model','supplier_max_price','GulongPH_slashed_backend','GulongPH_slashed_adjusted']].set_index('model').style.format(precision = 2))
                with r2:
                  st.dataframe(df_temmmp_.loc[df_temmmp_['GulongPH_backend']!=df_temmmp_['GulongPH_adjusted'],['model','supplier_max_price','GulongPH_backend','GulongPH_adjusted']].set_index('model').style.format(precision = 2))
                with r3:
                  st.dataframe(df_temmmp_.loc[df_temmmp_['b2b_backend']!=df_temmmp_['b2b_adjusted'],
                                              ['model','supplier_max_price','b2b_backend','b2b_adjusted']].set_index('model').style.format(precision = 0))
                with r4:
                  st.dataframe(df_temmmp_.loc[df_temmmp_['marketplace_backend']!=df_temmmp_['marketplace_adjusted'],['model','supplier_max_price','marketplace_backend','marketplace_adjusted']].set_index('model').style.format(precision = 0))


            final_df.columns = final_columns
            if edit_mode =='Manual':
                for c in range(len(col_tier)):
                    final_df[col_tier[c]] = final_df['Supplier_Price'].apply(lambda x: mp.consider_GP(x,col_GP[c]))
                final_df['3+1_promo_per_tire'] = final_df['Supplier_Price'].apply(lambda x: mp.promotize(x,GP_promo))
                st.info('Make sure that the cells that have been changed are included in the selected cells in the pivot table.\nMake sure that the checkbox for "Show captured erroneous values only" is unchecked.')
            

            #df_xlsx = to_excel(final_df)
            st.download_button(label='📥 Download Current Result',
                                            data= convert_df(final_df) ,
                                            file_name= 'website_template.csv')

            st.sidebar.caption('Last updated on 2023/04/12')

            st.markdown("""
                        ---
                        """)
    
    elif chosen_tab == '2':
        with placeholder:
            # supplier files upload
            if 'df_supplier' not in st.session_state:
                st.session_state['df_supplier'] = None
            
            with st.expander('Supplier Files Upload', expanded = False):
                df_supplier = st_wrapper_catalog.main()
                
                if df_supplier is not None:
                    st.session_state['df_supplier'] = df_supplier
                    
                else:
                    df_supplier = st.session_state['df_supplier']
                    #df_supplier = None
                
                if st.session_state['df_supplier'] is not None:
                    st.write(st.session_state['df_supplier'])
            
            # select columns to show
            with st.expander('Include/remove columns in list:'):
                beta_multiselect = st.container()
                check_all = st.checkbox('Select all', value=False)
                
                supplier_list = st_wrapper_catalog.get_supplier_names()
                
                # list of default columns to show
                if df_supplier is not None:
                    def_list = supplier_list if check_all else list(df_supplier.supplier.unique())
                else:
                    def_list = []
                    
                selected_cols = beta_multiselect.multiselect('Included columns in table:',
                                               options = supplier_list,
                                               default = def_list)
                selected_cols = list(set(selected_cols))
            
            df_show_cols = ['model', 'make', 'pattern', 'dimensions', 'GoGulong',
                            'TireManila', 'PartsPro', 'qty_tiremanila', 'year', 
                            'activity']
            df_show = data_dict['df_final'][df_show_cols]
                                           
            # merge supplier df if uploaded files
            if df_supplier is not None:
                qty_supp = [f'qty_{s}' for s in list(df_supplier.supplier.unique()) \
                            if f'qty_{s}' in df_supplier.columns]
                price_supp = [f'price_{s}' for s in list(df_supplier.supplier.unique()) \
                              if f'price_{s}' in df_supplier.columns]
                supplier_cols = ['similar_pattern', 'correct_specs',
                                 'brand'] + qty_supp + price_supp
                
                df_show = df_show.merge(df_supplier[supplier_cols],
                                        how = 'right',
                                        left_on = ['pattern', 'dimensions', 
                                                   'make'],
                                        right_on = ['similar_pattern', 'correct_specs', 
                                                    'brand'],
                                        suffixes = ('', '_')).drop_duplicates()
                df_show = df_show.dropna(how='all', axis=0, ignore_index = True)
                selected_cols.extend(qty_supp + price_supp)
            
                qty_cols = [c for c in df_show.columns if 'qty_' in c]
                df_show['preorder'] = df_show.apply(lambda x: preorder_calc(x[qty_cols]), axis=1)
                
                # build and show table
                drop_cols = ['brand', 'correct_specs']
                df_show = df_show.drop(labels = drop_cols,
                                       axis = 1)
                
                response = build_grid(df_show)
               
                supp_cols = st.columns([3,2])
                with supp_cols[0]:
                    st.write(f"""Results: 
                                 {len(response['data'])}/{len(df_show)} entries""")
                    
                with supp_cols[1]:
                    st.download_button(label="📥 Download this table.",
                                        data=convert_df(pd.DataFrame.from_dict(response['data'])),
                                        file_name='stocks_check.csv',
                                        mime='text/csv')

                    
    elif chosen_tab == '3':
        with placeholder:
            url = st.text_input('Enter Google Sheet URL')
            
            confirm_btn = st.button('Confirm')
            
            if len(url) and confirm_btn:
            
                df = gulong_sku_match.import_sheet(url, creds)
                
                df_data = gulong_sku_match.clean_data(df, data_dict['df_final'])
                
                df_merged = gulong_sku_match.match_df(df_data, data_dict['df_final'])
                
                st.dataframe(df_merged)
                
                if len(df_merged):
                    st.download_button('Download table',
                                       data = convert_df(df_merged),
                                       file_name = 'gulong_matches.csv',
                                       key = 'gulong_match')
    else:
        pass
    
    st.markdown("---")
    
    

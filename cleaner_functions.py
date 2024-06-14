# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:46:11 2024

@author: carlo
"""
import pandas as pd
import numpy as np
import re
from decimal import Decimal
from fuzzywuzzy import process, fuzz

def fix_names(sku_name, comp=None):
    '''
    Fix product names to match competitor names
    
    Parameters
    ----------
    sku_name: str
        input SKU name string
    comp: list (optional)
        optional list of model names to compare with
    
    Returns
    -------
    name: str
        fixed names as UPPERCASE
    '''
    
    # replacement should be all caps
    change_name_dict = {'TRANSIT.*ARZ(.|\s)?6-X' : 'TRANSITO ARZ 6-X', #ARIVO
                        'TRANSIT.*ARZ(.|\s)?6-A' : 'TRANSITO ARZ 6-A', #ARIVO
                        'TRANSIT.*ARZ(.|\s)?6-M' : 'TRANSITO ARZ 6-M', #ARIVO
                        'TRANSIT.*ARZ(.|\s)?6-C' : 'TRANSITO ARZ 6-C', #ARIVO
                        'OPA25': 'OPEN COUNTRY A25', # TOYO
                        'OPA28': 'OPEN COUNTRY A28', # TOYO
                        'OPA32': 'OPEN COUNTRY A32', # TOYO
                        'OPA33': 'OPEN COUNTRY A33', # TOYO
                        'OPAT\+': 'OPEN COUNTRY AT PLUS',# TOYO
                        'OPAT2': 'OPEN COUNTRY AT 2', # TOYO
                        'OPMT2': 'OPEN COUNTRY MT 2', # TOYO
                        'OPAT OPMT': 'OPEN COUNTRY AT', # TOYO
                        'OPAT': 'OPEN COUNTRY AT', # TOYO
                        'OPMT': 'OPEN COUNTRY MT', # TOYO
                        'OPRT': 'OPEN COUNTRY RT', # TOYO
                        'OPUT': 'OPEN COUNTRY UT', # TOYO
                        'DC -80': 'DC-80', #DOUBLECOIN
                        re.escape('DC -80+') : 'DC-80+', #DOUBLECOIN
                        'KM3': 'MUD-TERRAIN T/A KM3', # BFGOODRICH
                        'KO2': 'ALL-TERRAIN T/A KO2', # BFGOODRICH
                        'TRAIL-TERRAIN T/A' : 'TRAIL-TERRAIN', # BFGOODRICH
                        '265/70/R16 GEOLANDAR 112S': 'GEOLANDAR A/T G015',
                        '265/65/R17 GEOLANDAR 112S' : 'GEOLANDAR A/T G015',
                        '265/65/R17 GEOLANDAR 112H' : 'GEOLANDAR G902',
                        'GEOLANDAR A/T 102S': 'GEOLANDAR A/T-S G012',
                        'GEOLANDAR A/T': 'GEOLANDAR A/T G015',
                        'ASSURACE MAXGUARD SUV': 'ASSURANCE MAXGUARD SUV', #GOODYEAR
                        'EFFICIENTGRIP SUV': 'EFFICIENTGRIP SUV', #GOODYEAR
                        'EFFICIENGRIP PERFORMANCE SUV':'EFFICIENTGRIP PERFORMANCE SUV', #GOODYEAR
                        'WRANGLE DURATRAC': 'WRANGLER DURATRAC', #GOODYEAR
                        'WRANGLE AT ADVENTURE': 'WRANGLER AT ADVENTURE', #GOODYEAR
                        'WRANGLER AT ADVENTURE': 'WRANGLER AT ADVENTURE', #GOODYEAR
                        'WRANGLER AT SILENT TRAC': 'WRANGLER AT SILENTTRAC', #GOODYEAR
                        'ENASAVE EC300+': 'ENSAVE EC300 PLUS', #DUNLOP
                        'SAHARA AT2' : 'SAHARA AT 2',
                        'SAHARA MT2' : 'SAHARA MT 2',
                        'POTENZA RE003 ADREANALIN': 'POTENZA RE003 ADRENALIN', #BRIDGESTONE
                        'POTENZA RE004': 'POTENZA RE004', #BRIDGESTONE
                        'SPORT MAXX 050' : 'SPORT MAXX 050', #DUNLOP
                        'DUELER H/T 470': 'DUELER H/T 470', # BRIDGESTONE
                        'DUELER H/T 687': 'DUELER H/T 687 RBT', # BRIDGESTONE
                        'DUELER A/T 697': 'DUELER A/T 697', # BRIDGESTONE
                        'DUELER A/T 693': 'DUELER A/T 693 RBT', # BRIDGESTONE
                        'DUELER H/T 840' : 'DUELER H/T 840 RBT', # BRIDGESTONE
                        'EVOLUTION MT': 'EVOLUTION M/T', #COOPER
                        'BLUEARTH AE61' : 'BLUEARTH XT AE61', #YOKOHAMA
                        'BLUEARTH ES32' : 'BLUEARTH ES ES32', #YOKOHAMA
                        'BLUEARTH AE51': 'BLUEARTH GT AE51', #YOKOHAMA
                        'COOPER STT PRO': 'STT PRO',
                        'COOPER AT3 LT' : 'AT3 LT',
                        'COOPER AT3 XLT' : 'AT3 XLT',
                        'A/T3' : 'AT3',
                        'ENERGY XM2+' : 'ENERGY XM2+',
                        'ENERGY XM2' : 'ENERGY XM2',
                        re.escape('ENERGY XM+') : 'ENERGY XM2+',
                        re.escape('XM2+') : 'ENERGY XM2+',
                        'AT3 XLT': 'AT3 XLT',
                        'ADVANTAGE T/A DRIVE' : 'ADVANTAGE T/A DRIVE',
                        'ADVANTAGE T/A SUV' : 'ADVANTAGE T/A SUV',
                        'AGILIS 3' : 'AGILIS 3',
                        'PRIMACY 4 ST' : 'PRIMACY 4 ST'
                        }
    
    if pd.isna(sku_name) or (sku_name is None):
        return np.NaN
    
    else:
        # uppercase and remove double spaces
        raw_name = re.sub('  ', ' ', sku_name).upper().strip()
        # specific cases
        for key in change_name_dict.keys():
            if re.search(key, raw_name):
                return change_name_dict[key]
            else:
                continue
        
        # if match list provided
        
        if comp is not None:
            # check if any name from list matches anything in sku name
            match_list = [n for n in comp if re.search(n, raw_name)]
            # exact match from list
            if len(match_list) == 1:
                return match_list[0]
            # multiple matches (i.e. contains name but with extensions)
            elif len(match_list) > 1:
                long_match = ''
                for m in match_list:
                    if len(m) > len(long_match):
                        long_match = m
                return long_match
            # no match
            else:
                return raw_name
        else:
            return raw_name

def combine_specs(w, ar, d, mode = 'SKU'):
    '''
    
    Parameters
    - w: string
        section_width
    - ar: string
        aspect_ratio
    - d: string
        diameter
    - mode: string; optional
        SKU or MATCH
    
    Returns
    - combined specs with format for SKU or matching
    
    >>> combine_specs('175', 'R', 'R15', mode = 'SKU')
    '175/R15'
    >>> combine_specs('175', '65', 'R15', mode = 'SKU')
    '175/65/R15'
    >>> combine_specs('33', '12.5', 'R15', mode = 'SKU')
    '33X12.5/R15'
    >>> combine_specs('LT175', '65', 'R15C', mode = 'SKU')
    'LT175/65/R15C'
    >>> combine_specs('LT175', '65', 'R15C', mode = 'MATCH')
    '175/65/15'
    >>> combine_specs('175', '65', '15', mode = 'SKU')
    '175/65/R15'
    
    '''
    w, ar, d = str(w), str(ar), str(d)
    
    if mode == 'SKU':
        # formatting for diameter
        d = d if 'R' in d else 'R' + d 
        # formatting for aspect ratio
        if ar != 'R':
            if '.' in ar:
                return w + 'X' + ar + '/' + d
            else:
                return '/'.join([w, ar, d])
        else:
            return w + '/' + d
            
    elif mode == 'MATCH':
        w = ''.join(re.findall('[0-9]|\.', str(w)))
        ar = ''.join(re.findall('[0-9]|\.|R', str(ar)))
        d = ''.join(re.findall('[0-9]|\.', str(d)))
        return '/'.join([w, ar, d])

    else:
        combine_specs(str(w), str(ar), str(d), mode = 'SKU')

def remove_trailing_zero(num):
    '''
    Removes unnecessary zeros from decimals

    Parameters
    ----------
    num : Decimal(number)
        number applied with Decimal function (see import decimal from Decimal)

    Returns
    -------
    number: Decimal
        Fixed number in Decimal form

    '''
    return num.to_integral() if num == num.to_integral() else num.normalize()

def clean_width(w, model = None):
    '''
    Clean width values
    
    Parameters
    ----------
    d: string
        width values in string format
        
    Returns:
    --------
    d: string
        cleaned diameter values
    
    DOCTESTS:
    >>> clean_width('7')
    '7'
    >>> clean_width('175')
    '175'
    >>> clean_width('6.50')
    '6.5'
    >>> clean_width('27X')
    '27'
    >>> clean_width('LT35X')
    'LT35'
    >>> clean_width('8.25')
    '8.25'
    >>> clean_width('P265.5')
    'P265.5'
    >>> clean_width(np.NaN)
    nan
    
    '''
    if pd.notna(w):
        w = clean_specs(w)[0].strip().upper()
        # detects if input has expected format
        prefix_num = re.search('[A-Z]*[0-9]+.?[0-9]*(?=X|/|\s)?', w)
        if prefix_num is not None:
            num_str = ''.join(re.findall('[0-9]|\.', prefix_num[0]))
            num = str(remove_trailing_zero(Decimal(num_str)))
            prefix = w.split(num_str)[0]
            return prefix + num
        else:
            return np.NaN
    else:
        if model is None:
            return np.NaN
        else:
            try:
                width = model.split('/')[0].split(' ')[-1].strip().upper()
                return clean_width(width)   
            except:
                return np.NaN

def clean_aspect_ratio(ar, model = None):
    
    '''
    Clean raw aspect ratio input
    
    Parameters
    ----------
    ar: float or string
        input raw aspect ratio data
    model: string; optional
        input model string value of product
        
    Returns
    -------
    ar: string
        fixed aspect ratio data in string format for combine_specs
    
    DOCTESTS:
    >>> clean_aspect_ratio('/')
    'R'
    >>> clean_aspect_ratio('.5')
    '9.5'
    >>> clean_aspect_ratio('14.50')
    '14.5'
    >>> clean_aspect_ratio(np.NaN)
    'R'
    
    '''
    error_ar = {'.5' : '9.5',
                '0.': '10.5',
                '2.': '12.5',
                '3.': '13.5',
                '5.': '15.5',
                '70.5': '10.5'}
    
    if pd.notna(ar):
        #ar = re.sub('/', '', str(ar)).strip()
        
        # aspect ratio is faulty
        if str(ar) in ['0', 'R1', '/', 'R']:
            result = 'R'
        # incorrect parsing osf decimal aspect ratios
        elif str(ar) in error_ar.keys():
            result = error_ar[str(ar)]
        # numeric/integer aspect ratio
        elif str(ar).isnumeric():
            result = str(ar)
        # alphanumeric
        elif str(ar).isalnum():
            result = ''.join(re.findall('[0-9]', str(ar)))
        
        # decimal aspect ratio with trailing 0
        elif '.' in str(ar):
            #return str(remove_trailing_zero(Decimal(str(ar))))
            result = str(ar).rstrip('0')
        
        else:
            result = 'R'
            
        if any((s := sep) in ar for sep in ['/', 'X']):
            temp = result.split(s)
            if len(temp) > 1:
                result = temp[1]
            else:
                result = temp[0]
        else:
            pass
          
    else:
        result = 'R'
    
    return result

def clean_diameter(d):
    '''
    Fix diameter values
    
    Parameters
    ----------
    d: string
        diameter values in string format
        
    Returns:
    --------
    d: string
        fixed diameter values
    
    DOCTESTS:
    >>> clean_diameter('R17LT')
    'R17LT'
    >>> clean_diameter('R22.50')
    'R22.5'
    >>> clean_diameter('15')
    'R15'
    >>> clean_diameter(np.NaN)
    nan
    
    '''
    if pd.notna(d):
        d = str(d).strip().upper()
        num_suffix = re.search('[0-9]+.?[0-9]*[A-Z]*', d)
        if num_suffix is not None:
            num_str = ''.join(re.findall('([0-9]|\.)', num_suffix[0]))
            num = str(remove_trailing_zero(Decimal(num_str)))
            suffix = num_suffix[0].split(num_str)[-1]
            return f'R{num}{suffix}'
        else:
            return np.NaN
        
    else:
        return np.NaN



def clean_speed_rating(sp):
    '''
    Clean speed rating of gulong products
    
    DOCTESTS:
    >>> clean_speed_rating('W XL')
    'W'
    >>> clean_speed_rating('0')
    'B'
    >>> clean_speed_rating('118Q')
    'Q'
    >>> clean_speed_rating('T/H')
    'T'
    >>> clean_speed_rating('-')
    nan
    
    '''

    # not NaN
    if pd.notna(sp):
        # baseline correct
        sp = sp.strip().upper()
        # detect if numerals are present 
        num = re.search('[0-9]{2,3}', sp)
        
        if num is None:
            pass
        else:
            # remove if found
            sp = sp.split(num[0])[-1].strip()
            
        if 'XL' in sp:
            return sp.split('XL')[0].strip()
        elif '/' in sp:
            return sp.split('/')[0].strip()
        elif sp == '0':
            return 'B'
        elif sp == '-':
            return np.NaN
        else:
            return sp
    else:
        return np.NaN


def combine_sku(make, w, ar, d, model, load, speed):
    '''
    DOCTESTS:
            
    >>> combine_sku('ARIVO', '195', 'R', 'R15', 'TRANSITO ARZ 6-X', '106/104', 'Q')
    'ARIVO 195/R15 TRANSITO ARZ 6-X 106/104Q'
    
    '''
    specs = combine_specs(w, ar, d, mode = 'SKU')
    
    try:
        SKU = ' '.join([make, specs, model])
    except:
        SKU = ' '.join([make, specs])
    
    finally:
        if (load in ['nan', np.NaN, None, '-', '']) or \
            (speed in ['nan', np.NaN, None, '', '-']):
            pass
        else:
            SKU = SKU + ' ' + load + speed
        return SKU

def clean_specs(x):
    '''
    Extracts cleaned tire specs information from product title
    '''
    if pd.isna(x):
        return ''
    else:
        # baseline correction
        x = x.upper().strip()
        if ((match := re.search('[0-9]+(\.)?[0-9]*(X|(\/)*|\s)?([0-9]+(\.)?[0-9]*)?(\s)*((\/)*|\s|\-)?[A-Z]*(\s)*[0-9]*([A-Z]+)?', x)) is not None):
        #if ((match := re.search('[0-9]+(\.)?[0-9]*(X|(\/)*|\s)?([0-9]+(\.)?[0-9]*)?(\s)*((\/)*|\s|\-)?[A-Z]*(\s)*[0-9]+([A-Z]+)?', x)) is not None):
            #specs =  [num[0] for n in re.split('X|Z?R|/|-', match[0]) if (num := re.search('[0-9]+(.)?[0-9]+', n.strip())) is not None]
            specs =  [num[0] for n in re.split('X|Z?R|/|-', match[0]) if (num := re.search('[0-9]+(.)?[0-9]*', n.strip())) is not None]
            if len(specs) == 3:
                if '.' in specs[1]:
                    specs[1] = format(float(specs[1]), '.2f')
            elif len(specs) == 2:
                specs.insert(1, 'R')

        else:
            specs = ['']*3
        
        return specs

def clean_tire_size(s : str) -> tuple:
    '''
    Extracts width, aspect ratio, and diameter information from tire size string
    
    DOCTESTS:
    >> clean_tire_size('195/55/R15')
    (195, 55, 15)
    
    > clean_tire_size('195/55 R15')
    '''
    #s = re.search('[0-9]+(X|/)[0-9]+(.)?([0-9]+)?(\s+)?[A-Z]+(\s+)?(\/)?[0-9]+([A-Z]+)?', s)
    sizes_list = clean_specs(s)
    
    try:
        if sizes_list[0] is not None:
            w = clean_width(sizes_list[0])
        else:
            w = clean_width(re.search("(\d{2,3})|(\d{2,3}[Xx])|(\d{2,3} )", s[0])[0])
    except:
        w = None
    try:
        if sizes_list[1] is not None:
            ar = clean_aspect_ratio(sizes_list[1])
        else:
            ar = clean_aspect_ratio(re.search("(?<=/).*(?=R)|(?<=[Xx]).*(?=R)|( R)", s[0])[0].strip())
    except:
        ar = 'R'
    try:
        if sizes_list[2] is not None:
            d = clean_diameter(sizes_list[2])
        else:
            d = clean_diameter(re.search('R.*', s[0])[0].replace(' ', ''))
    except:
        d = None
    
    return w, ar, d



def clean_price(x : str) -> str:
    '''
    Cleans price string values from scraped entries
    '''
    if pd.isna(x):
        return np.NaN
    else:
        # baseline correct
        x = str(x).upper().strip()
        # normal result
        try:
            if 'Million' in x or 'M' in x:
                match = re.search('[1-9](.)?[0-9]+((?<!MILLION)|(?<!M))', x)
                return str(float(match[0])*1E6)
            else:
                match = re.search('[1-9]+(,)?[0-9]+(,)?[0-9]+(.)?[0-9]+',x)
                return ''.join(match[0].strip().split(','))
        # unexpected result
        except:
            # get all digits
            try:
                return ''.join(re.findall('[0-9]', x))
            # return cleaned string
            except:
                return x

def import_makes():
    '''
    Import list of makes
    '''
    with open("gulong_makes.txt") as makes_file:
        makes = makes_file.readlines()
        
    makes = [re.sub('\n', '', m).strip() for m in makes]
    return makes

def clean_makes(x : str, 
                ref : None) -> str:
    # x = x.strip().upper()
    # if any((match := m) for m in makes if lev_dist(m, x) <= 1):
    #     return match
    # else:
    #     return x.split(' ')[0]
    match = process.extractOne(x, ref.brand.unique())
    if match:
        return match[0]
    else:
        return x.split(' ')[0]

def clean_make(x, makes, model = None):
    '''
    Helper function to cleanup gulong makes
    
    Parameters:
        - x: string; value of gulong make, can be NaN
        - makes: list; reference list of gulong makes from gulong_makes.txt
    
    Output:
        cleaned string value of gulong make
    '''
    if pd.notna(x):
        # baseline string correction if not NaN
        x = x.strip().upper()
        # check if partial match exists
        if any((match := m) for m in makes if fuzz.partial_ratio(m, x) >= 95):
            return match
        # if model is provided, check model if make is included
        elif model is not None and any((match := m) for m in makes if fuzz.partial_ratio(m, model.upper().strip()) >= 95):
            return match
        # check for best match given threshold
        elif process.extractOne(x, makes)[1] >= 75:
            return process.extractOne(x, makes)[0]
        else:
            return x
    else:
        # if input is NaN
        return np.NaN

def clean_model(x : str, 
                ref : None,
                make : str = None):
    '''
    Extracts cleaned tire model from product title
    '''
    if pd.isna(x):
        return np.NaN
    else:
        # baseline correction
        x = x.upper().strip()
        
        # make
        try:
            if make is not None:
                pass
            else:
                make = clean_makes(x, ref)
        except:
            make = ''
        # tire specs
        try:
            #tire_specs = clean_specs(x)
            tire_specs = re.search('[0-9]+(\.)?[0-9]*(X|\/|\s)?([0-9]+(\.)?[0-9]*)?(\s)*(\/|\-|\s)?R?(\s)?[0-9]+([A-Z]+)?', x)[0]
            #tire_specs = re.search('[0-9]+(X|/)[0-9]+(.)?([0-9]+)?(\s+)?[A-Z]+(\s+)?[0-9]+([A-Z]+)?', x)[0]
        except:
            tire_specs = ''
        
        #ply rating
        try:
            ply = re.search('[0-9]{1,2}PR', x)[0]
        except:
            ply = ''
        
        # load speed index
        try:
            load_speed_index = re.search('[0-9]{2,3}(\/[0-9]{2,3})?[A-Z]', x)[0]
        except:
            load_speed_index = ''
        
        for element in [make, tire_specs, ply, load_speed_index]:
            x = re.sub(element, '', x).strip()
        
        x = re.sub("TIRES", "", x).strip()
            
        return x
    
def clean_year(y : str or int) -> [np.NaN, str]:
    '''
    Cleans input year to resolve out of range values
    '''
    if pd.isna(y) or (y is None):
        result = np.NaN
    
    else:
        str_num = re.sub("'", "", str(y)).strip()
        if len(str_num) == 4:
            if 10 <= int(str_num[-2:]) < 50:
                result = '20' + str_num[-2:]
            
            elif 50 <= int(str_num[-2:]) < 99:
                result = '19' + str_num[-2:]
            
            else:
                result = '20' + str_num[1] + str_num[3]                

        elif (len(str_num) == 2) and (int(str_num) > 0):
            result = '20' + str_num
        
        else:
            result = np.NaN
    
    return result
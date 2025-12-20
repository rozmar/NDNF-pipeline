import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build    
import pandas as pd
#import matplotlib.pyplot as plt
#import numpy as np
import os
import numpy as np# use creds to create a client to interact with the Google Drive API
import json




def create_client(google_creds_json):
    scope = ['https://www.googleapis.com/auth/analytics.readonly',
            'https://www.googleapis.com/auth/drive',
            'https://www.googleapis.com/auth/spreadsheets',
            ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(google_creds_json, scope)
    client = gspread.authorize(creds)
    return creds,client

#%% open 

def fetch_lastmodify_time(spreadsheetname,client,creds):
    modifiedtime = None
    ID = None
    service = build('drive', 'v3', credentials=creds)
    wb = client.open(spreadsheetname)
    ID = wb.id
    if ID:
        modifiedtime = service.files().get(fileId = ID,fields = 'modifiedTime').execute()
    return modifiedtime

def fetch_sheet_titles(spreadsheetname,client):
    wb = client.open(spreadsheetname)
    sheetnames = list()
    worksheets = wb.worksheets()
    for sheet in worksheets:
        sheetnames.append(sheet.title)
    return sheetnames

def fetch_sheet(spreadsheet_name,sheet_title,client):
    #%%
    wb = client.open(spreadsheet_name)
    sheetnames = list()
    worksheets = wb.worksheets()
    for sheet in worksheets:
        sheetnames.append(sheet.title)
    if sheet_title in sheetnames:
        print(sheet_title)
        idx_now = sheetnames.index(sheet_title)
        if idx_now > -1:
            params = {'majorDimension':'ROWS'}
            temp = wb.values_get(sheet_title+'!A1:OO10000',params)
            temp = temp['values']
            header = temp.pop(0)
            data = list()
            for row in temp:
                data.append(row)
            df = pd.DataFrame(data, columns = header)
            return df
        else:
            return None
    else:
        return None

def update_metadata(notebook_name,metadata_dir, google_creds_json): 
    # TODO: This function should also save the current version of the 
    # spreadsheet with date and time so changes over time are logged
    """
    Docstring for update_metadata
    
    :param notebook_name: name of the google spreadsheet notebook
    :param metadata_dir: directory to store metadata files
    :param google_creds_json: path to the Google credentials JSON file
    """
    creds, client = create_client(google_creds_json)
    lastmodify = fetch_lastmodify_time(notebook_name,client,creds)
    last_modify_file_name = notebook_name.replace(' ','_')+'_'+'last_modify_time.json'
    if last_modify_file_name in os.listdir(metadata_dir):
        with open(os.path.join(metadata_dir,last_modify_file_name)) as timedata:
            lastmodify_prev = json.loads(timedata.read())
        redownload = lastmodify != lastmodify_prev
    else:
        redownload = True
    if redownload:
        print('updating metadata from google drive')
        sessions = fetch_sheet_titles(notebook_name, client)
        for session in sessions:
            df_wr = fetch_sheet(notebook_name,session,client)
            if type(df_wr) == pd.DataFrame:
                df_wr.to_csv(os.path.join(metadata_dir,'{}_{}.csv'.format(notebook_name,session))) 
        with open(os.path.join(metadata_dir,last_modify_file_name), "w") as write_file:
            json.dump(lastmodify, write_file)
        print('metadata updated')
    else:
        print('metadata is already up to date')

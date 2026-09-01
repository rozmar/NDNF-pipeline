import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
import pandas as pd
#import matplotlib.pyplot as plt
#import numpy as np
import os
import numpy as np# use creds to create a client to interact with the Google Drive API
import json
import hashlib
import random
import time
from datetime import datetime, timezone
from pathlib import Path

# Google API status codes worth retrying (transient server-side issues, not
# auth/permission/not-found errors, which won't be fixed by retrying).
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_retryable_api_error(exc):
    if not isinstance(exc, gspread.exceptions.APIError):
        return False
    try:
        status = exc.response.status_code
    except AttributeError:
        status = None
    return status in _RETRYABLE_STATUS_CODES


def _call_with_retry(func, *args, max_retries=5, base_delay=2, **kwargs):
    """
    Call func(*args, **kwargs), retrying with exponential backoff on
    transient Google API errors (e.g. 503 Service Unavailable) instead of
    letting them bubble up and hang/crash the script.
    """
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except gspread.exceptions.APIError as exc:
            if attempt == max_retries or not _is_retryable_api_error(exc):
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f'Google API error ({exc}), retrying in {delay:.1f}s '
                  f'(attempt {attempt + 1}/{max_retries})')
            time.sleep(delay)


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
    wb = _call_with_retry(client.open, spreadsheetname)
    ID = wb.id
    if ID:
        modifiedtime = _call_with_retry(
            service.files().get(fileId=ID, fields='modifiedTime').execute)
    return modifiedtime

def fetch_sheet_titles(spreadsheetname,client):
    wb = _call_with_retry(client.open, spreadsheetname)
    sheetnames = list()
    worksheets = _call_with_retry(wb.worksheets)
    for sheet in worksheets:
        sheetnames.append(sheet.title)
    return sheetnames

def fetch_sheet(spreadsheet_name,sheet_title,client):
    #%%
    wb = _call_with_retry(client.open, spreadsheet_name)
    sheetnames = list()
    worksheets = _call_with_retry(wb.worksheets)
    for sheet in worksheets:
        sheetnames.append(sheet.title)
    if sheet_title in sheetnames:
        print(sheet_title)
        idx_now = sheetnames.index(sheet_title)
        if idx_now > -1:
            params = {'majorDimension':'ROWS'}
            temp = _call_with_retry(wb.values_get, sheet_title+'!A1:OO10000', params)
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

def _hash_sheet_values(values):
    return hashlib.sha256(json.dumps(values).encode('utf-8')).hexdigest()

def update_metadata(notebook_name, metadata_dir, google_creds_json):
    """
    Function to update metadata from a google spreadsheet notebooks.

    Change detection is done per-sheet, by hashing the actual fetched values,
    rather than by comparing the spreadsheet's Drive `modifiedTime`. Google
    Forms writing responses into a linked sheet tab does not reliably bump
    that timestamp, so gating on it silently misses form-submitted rows.

    :param notebook_name: name of the google spreadsheet notebook
    :param metadata_dir: directory to store metadata files
    :param google_creds_json: path to the Google credentials JSON file
    """
    _, client = create_client(google_creds_json)

    # Open spreadsheet once and reuse
    wb = _call_with_retry(client.open, notebook_name)

    # Get all sheet titles in one call, then fetch all data in one batch call
    worksheets = _call_with_retry(wb.worksheets)
    sheet_titles = [ws.title for ws in worksheets]
    ranges = [f"{title}!A1:OO10000" for title in sheet_titles]
    batch_result = _call_with_retry(
        wb.values_batch_get, ranges, params={'majorDimension': 'ROWS'})

    hashes_file_name = notebook_name.replace(' ', '_') + '_sheet_hashes.json'
    hashes_path = os.path.join(metadata_dir, hashes_file_name)
    if os.path.exists(hashes_path):
        with open(hashes_path) as f:
            previous_hashes = json.load(f)
    else:
        previous_hashes = {}

    current_hashes = dict(previous_hashes)
    archive_fname = None
    for title, value_range in zip(sheet_titles, batch_result.get('valueRanges', [])):
        values = value_range.get('values', [])
        if not values:
            continue
        sheet_hash = _hash_sheet_values(values)
        if previous_hashes.get(title) == sheet_hash:
            continue

        print(f'{title} changed, updating metadata')
        if archive_fname is None:
            archive_fname = '{}.csv'.format(
                datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S'))

        df = pd.DataFrame(values[1:], columns=values[0])
        df.to_csv(os.path.join(metadata_dir, f'{notebook_name}_{title}.csv'))
        archive_path = os.path.join(metadata_dir, 'archive', f'{notebook_name}_{title}')
        Path(archive_path).mkdir(parents=True, exist_ok=True)
        df.to_csv(os.path.join(archive_path, archive_fname))

        current_hashes[title] = sheet_hash

    if current_hashes == previous_hashes:
        print('metadata is already up to date')
        return

    with open(hashes_path, 'w') as f:
        json.dump(current_hashes, f)
    print('metadata updated')

from __future__ import print_function
import httplib2
import os
import io
import numpy as np
import pandas as pd
import six
import sys
from apiclient import errors
from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage
from apiclient.http import MediaFileUpload, MediaIoBaseDownload

try:
    import argparse
    flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
except ImportError:
    flags = None

# If modifying these scopes, delete your previously saved credentials
# at ~/.credentials/drive-python-quickstart.json
SCOPES = 'https://www.googleapis.com/auth/drive'
CLIENT_SECRET_FILE = 'model/google_ocr/credentials.json'
APPLICATION_NAME = 'Drive API Python Quickstart'
if len(sys.argv)<2:
    sys.exit("Format: python model/google_ocr/test_googleocr.py <path_to_test_file>")

path_to_testfile = sys.argv[1]
def get_credentials():
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    """
    credential_path = os.path.join("./model/google_ocr", 'drive-python-quickstart.json')
    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        if flags:
            credentials = tools.run_flow(flow, store, flags)
        else:  # Needed only for compatibility with Python 2.6
            credentials = tools.run(flow, store)
        print('Storing credentials to ' + credential_path)
    return credentials

def main():
    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    service = discovery.build('drive', 'v3', http=http)
    ct = 0
    l=0
    if not os.path.exists(os.getcwd()+"/model/google_ocr/gocr_test"):
        os.mkdir(os.getcwd()+"/model/google_ocr/gocr_test")
    with open(os.getcwd()+'/'+path_to_testfile) as f:
        lines = f.readlines()
        for line in lines:
            loc, annot = line.split(" ",1)[0] , line.split(" ",1)[1]

            imgfile = loc
            txtfile = os.getcwd()+'/model/google_ocr/'+'gocr_test/'+str(ct)+'.txt'
            
            if os.path.exists(txtfile)==False:
                mime = 'application/vnd.google-apps.document'
                res = service.files().create(
                    body={
                        'name': imgfile,
                        'mimeType': mime
                    },
                    media_body=MediaFileUpload(imgfile, mimetype=mime, resumable=True)
                ).execute()

                downloader = MediaIoBaseDownload(
                    io.FileIO(txtfile, 'wb'),
                    service.files().export_media(fileId=res['id'], mimeType="text/plain")
                )

                done = False
                while done is False:
                    status, done = downloader.next_chunk()

                service.files().delete(fileId=res['id']).execute()
                print("Done")
            else:
                with open(txtfile) as fx:
                    l = fx.readlines()
                if len(l)<3:
                    mime = 'application/vnd.google-apps.document'
                    res = service.files().create(
                        body={
                            'name': imgfile,
                            'mimeType': mime
                        },
                        media_body=MediaFileUpload(imgfile, mimetype=mime, resumable=True)
                    ).execute()

                    downloader = MediaIoBaseDownload(
                        io.FileIO(txtfile, 'wb'),
                        service.files().export_media(fileId=res['id'], mimeType="text/plain")
                    )

                    done = False
                    while done is False:
                        status, done = downloader.next_chunk()

                    service.files().delete(fileId=res['id']).execute()
                    print("Done")

            ct = ct+1
            print(ct)
            
                # file_handle = io.BytesIO()
                # downloader = MediaIoBaseDownload(
                #     file_handle,
                #     service.files().export_media(fileId=res['id'], mimeType="text/plain")
                # )
                # done = False
                # while done is False:
                #     status, done = downloader.next_chunk()
                # filevalue = file_handle.getvalue()
                # if not isinstance(filevalue, six.string_types):
                #     filevalue = filevalue.decode('UTF-8')
                
                # print(filevalue[21:])
                # pred_gocr.append(filevalue[21:])
                # service.files().delete(fileId=res['id']).execute()
                # # output = six.StringIO(filevalue)

                # print("Done.")


if __name__ == '__main__':
    main()

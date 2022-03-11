import pathlib
import pandas as pd
import dropbox
from dropbox.exceptions import AuthError

DROPBOX_ACCESS_TOKEN = 'uRy4tqw8s04AAAAAAAAAAcAuKjurfv8zR-IsK4qiaD2Wp7SCgz4u95FWBg6cRDF-'

def dropbox_connect():
    """Create a connection to Dropbox."""

    try:
        dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
    except AuthError as e:
        print('Error connecting to Dropbox with access token: ' + str(e))
    return dbx

def dropbox_list_files(path):
    """Return a Pandas dataframe of files in a given Dropbox folder path in the Apps directory.
    """

    dbx = dropbox_connect()

    try:
        files = dbx.files_list_folder(path).entries
        files_list = []
        for file in files:
            if isinstance(file, dropbox.files.FileMetadata):
                metadata = {
                    'name': file.name,
                    'path_display': file.path_display,
                    'client_modified': file.client_modified,
                    'server_modified': file.server_modified
                }
                files_list.append(metadata)

        df = pd.DataFrame.from_records(files_list)
        return df.sort_values(by='server_modified', ascending=False)

    except Exception as e:
        print('Error getting list of files from Dropbox: ' + str(e))



dbx = dropbox_connect()
files = dbx.files_list_folder('/AVL - Cotton Plant Phenotyping/Annotations/Datasets').entries
#dbx.users_get_current_account()
#for entry in dbx.files_list_folder('').entries:
#    print(entry.name)
#dropbox_list_files('/AVL - Cotton Plant Phenotyping/')
for file in files:
            print(isinstance(file, dropbox.files.FileMetadata))
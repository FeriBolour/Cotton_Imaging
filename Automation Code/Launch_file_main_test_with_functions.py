import os
from Editlaunchfile_function import editlaunchfile
from Defaultlaunchfile_function import defaultlaunchfile

# Edits launchfile name (user given name)
Filenames = editlaunchfile()

# Run scanning session in between

# Brings the edited launcfilename back to its default settings ('Choosename.db')
defaultlaunchfile(Filenames[0], Filenames[1], Filenames[2])
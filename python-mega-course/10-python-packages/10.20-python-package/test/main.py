# Placed outside of the root directory to import the installed package not the local one
# invoicing-pdf is the official "Name" of the package as it's listed on PyPI and used for installing
# But use "packages" name when IMPORTING
# Usually official name and practical name are the same but not always (i.e., sklearn)
from invoicing import invoice

invoice.generate()
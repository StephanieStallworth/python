from setuptools import setup

setup(
    name='invoicing-pdf', # * Your package will have this name; "official" name of package - users will install using this name; search pypi to make sure name is not already occupied
    packages=['invoicing'],  # * Name the package again; set as list in case there are multiple packages
    version='1.0.0',  # * To be increased every time your change your library; 1.0.1 = bug fix, 1.1.0 = non-breaking change, 2.0 = major breaking change (changing/removing argument names, etc)
    license='MIT',
    # Type of license. More here: https://help.github.com/articles/licensing-a-repository; means everybody is free to use this package
    description='This package can be used to convert Excel invoices to PDF invoices.',
    # Short description of your library
    author='Ardit Sulce',  # Your name
    author_email='your.email@example.com',  # Your email; dummy email is fine
    url='https://example.com',  # Homepage of your library (e.g. github or your website)
    keywords=['invoice', 'excel', 'pdf'],  # Keywords users can search on pypi.org
    install_requires=['pandas', 'fpdf', 'openpyxl'],  # Other 3rd-party libs that pip needs to install; THIRD PARTY dependencies that should be installed for package to work correctly (even if they're not being imported directly)
    classifiers=[
        'Development Status :: 3 - Alpha',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',  # Who is the audience for your library?
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',  # Type a license again
        'Programming Language :: Python :: 3.8',  # Python versions that your library supports
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='linex2metaspace',
    version='0.1.0',

    description='LINEX2 networks for metaspace data',
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://git.embl.de/trose/linex2_for_maldi',
    project_urls={  # Optional
        'Source': 'https://git.embl.de/trose/linex2_for_maldi',
        'Publication': "https://doi.org/10.1101/2022.02.04.479101"
    },

    author="Tim Daniel Rose",
    author_email="tim.rose@embl.de",

    license='AGPLv3',

    packages=find_packages(),
    install_requires=['linex2', 'networkx', "pandas", "matplotlib", "numpy"],
    python_requires=">=3.8",

    zip_safe=False,


    classifiers=[
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3"

    ]
)

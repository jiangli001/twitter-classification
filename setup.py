from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(name='tweet-classification',
      version='1.0',
      description='A Twitter issue classification project',
      long_description=long_description,
      author='Li Jiang',
      author_email='ljiang38@jhu.edu',
      packages=find_packages(where='src'),
      python_requires='>=3.5',
      extras_require={
          'text_processing': ['gensim', 'NLTK'],
          'machine_learning': ['scikit-learn>=0.23']
          'data_manipulation': ['pandas', 'numpy']
        },
      package_data={
        '': ['*.json', '*.xlsx']
        }
)
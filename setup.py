from setuptools import setup,find_packages

setup(name='man-opt',
      version='0.1',
      description='All codes for generating figures for the All-active manuscript',
      author='Ani Nandi',
      author_email='anin@alleninstitute.org',
      url="https://github.com/anirban6908/All-active-Manuscript",
      install_requires=[
        #'git+https://github.com/AllenInstitute/All-active-Workflow',
        'numpy>=1.6',
        'pandas>=0.18',
        'seaborn>=0.9.0',
        'scikit-learn>=0.20.3',
        'umap-learn>=0.3',
        'floweaver'],
      packages=find_packages(),
      platforms='any',
      python_requires='>=3.6'
    )

# -*- coding: utf-8 -*-
"""html2txt.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KBNKC_zALsArdvtLB6iBo1FehYsMqSq7
"""

from bs4 import BeautifulSoup
import os
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

!pip install inscriptis

from inscriptis import get_text

def convert_html_to_text(html_path, output_path):
  with open(html_path, "r") as f:
    html = f.read()

  text = get_text(html)

  with open(output_path, "w") as f:
    f.write(text)

# Replace with your desired directory in Google Drive
folder_path = '/content/drive/My Drive/grounding'

file_names = []
for filename in os.listdir(folder_path):
  file_names.append(filename)

for filename in file_names[1:]:
  html_path = os.path.join(folder_path, filename)
  output_path = os.path.join(folder_path, 'mytext',filename.replace(".html", ".txt"))
  convert_html_to_text(html_path, output_path)

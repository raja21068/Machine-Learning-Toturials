# You can read/learn more about latest updates about textract on their
# official documents site at http://textract.readthedocs.io/en/latest/

import textract
# Extracting text from normal pdf
text = textract.process('chapter5/Data/pdf/raw_text.pdf', language='eng')
print(text)

# Extracting text from two columned pdf
text = textract.process('chapter5/Data/pdf/two_column.pdf', language='eng')
# Extracting text from scanned text pdf
text = textract.process('chapter5/Data/pdf/ocr_text.pdf', method='tesseract',
language='eng')
# Extracting text from jpg
text = textract.process('chapter5/Data/jpg/raw_text.jpg', method='tesseract',
language='eng')
# Extracting text from audio file
text = textract.process('chapter5/wav/raw_text.wav', language='eng')

text = textract.process('chapter5/Data/jpg/raw_text.jpg', method='tesseract', language='eng')
print (text)

text = textract.process('chapter5/Data/wav/raw_text.wav', language='eng')
print ("raw_text.wav: ", text)

text = textract.process('chapter5/Data/wav/standardized_text.wav', language='eng')
print ("standardized_text.wav: ", text)


from img2table.document import Image

from img2table.document import PDF
from img2table.ocr import TesseractOCR

# Instantiation of the image
img = Image(src="C:\\Works\\ISD\\test\\pdf\\table-word.jpg")

# Table identification
img_tables = img.extract_tables()

# Result of table identification
img_tables

# [ExtractedTable(title=None, bbox=(10, 8, 745, 314),shape=(6, 3)),
#  ExtractedTable(title=None, bbox=(936, 9, 1129, 111),shape=(2, 2))]


# Instantiation of the pdf
pdf = PDF(src="C:\\Works\\ISD\\test\\pdf\\ast_sci_data_tables_sample.pdf")

# Instantiation of the OCR, Tesseract, which requires prior installation
ocr = TesseractOCR(lang="eng")

# Table identification and extraction
pdf_tables = pdf.extract_tables(ocr=ocr)

# We can also create an excel file with the tables
pdf.to_xlsx('tables.xlsx',
            ocr=ocr)
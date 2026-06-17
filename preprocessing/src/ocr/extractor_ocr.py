import os
import fitz
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from dotenv import load_dotenv

# Carga las variables del .env en la raíz del proyecto
load_dotenv()

# Ruta al ejecutable de Tesseract (se configura en el .env)
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH")

# Ruta a Poppler (se configura en el .env)
POPPLER_PATH = os.getenv("POPPLER_PATH")

# Idiomas: español primero, inglés como respaldo
# (documentos UdeA pueden tener términos en inglés)
IDIOMAS_OCR = "spa+eng"


def extraer_texto_ocr(pdf_path: str, dpi: int = 300) -> list[dict]:
    """
    Convierte cada página del PDF a imagen y aplica OCR.
    
    dpi=300 es el estándar para documentos: suficiente calidad sin ser lento.
    dpi=150 si necesitas velocidad y el texto es grande.
    dpi=400 si el texto es muy pequeño o la calidad del escaneo es mala.
    """
    paginas_texto = []
    
    # Convierte todo el PDF a imágenes (una imagen por página)
    imagenes = convert_from_path(
        pdf_path,
        dpi=dpi,
        poppler_path=POPPLER_PATH
    )
    
    for num_pag, imagen in enumerate(imagenes):
        # Preprocesa la imagen para mejorar el OCR
        imagen_procesada = preprocesar_imagen(imagen)
        
        # Aplica OCR
        texto = pytesseract.image_to_string(
            imagen_procesada,
            lang=IDIOMAS_OCR,
            config="--psm 3"  # psm 3 = detección automática de columnas
        )
        
        paginas_texto.append({
            "pagina": num_pag + 1,
            "texto": texto,
            "metodo": "ocr",
            "dpi_usado": dpi
        })
    
    return paginas_texto


def preprocesar_imagen(imagen: Image.Image) -> Image.Image:
    """
    Mejora la imagen antes del OCR.
    Convierte a escala de grises y aumenta el contraste.
    """
    # Escala de grises: el OCR funciona mejor sin color
    imagen_gris = imagen.convert("L")
    return imagen_gris
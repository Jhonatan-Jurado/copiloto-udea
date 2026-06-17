import re

def limpiar_texto(texto: str, fuente: str = "nativo") -> str:
    """
    Pipeline de limpieza de texto extraído de PDF.
    Aplica pasos progresivos según el origen del texto.
    """
    texto = limpiar_caracteres_raros(texto)
    texto = normalizar_espacios(texto)
    texto = eliminar_encabezados_pie_pagina(texto)
    texto = normalizar_puntuacion(texto)
    texto = eliminar_lineas_vacias_excesivas(texto)
    
    if fuente == "ocr":
        # Correcciones extra para errores comunes de OCR en español
        texto = corregir_errores_ocr_espanol(texto)
    
    return texto.strip()


def limpiar_caracteres_raros(texto: str) -> str:
    """Elimina caracteres no imprimibles y de control."""
    # Mantiene: letras, números, puntuación normal, espacios, saltos de línea
    texto = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', texto)
    # Reemplaza comillas tipográficas por comillas normales
    texto = texto.replace('\u201c', '"').replace('\u201d', '"')
    texto = texto.replace('\u2018', "'").replace('\u2019', "'")
    return texto


def normalizar_espacios(texto: str) -> str:
    """Normaliza espacios múltiples y tabulaciones."""
    # Reemplaza tabulaciones y espacios múltiples por un solo espacio
    texto = re.sub(r'[ \t]+', ' ', texto)
    # Elimina espacios al inicio/fin de cada línea
    texto = '\n'.join(linea.strip() for linea in texto.split('\n'))
    return texto


def eliminar_encabezados_pie_pagina(texto: str) -> str:
    """
    Elimina patrones repetitivos de encabezados y pies de página.
    Ajusta los patrones según los documentos reales de UdeA.
    """
    patrones_a_eliminar = [
        r'Universidad de Antioquia\s*\n',           # encabezado repetido
        r'Secretaría General\s*\n',
        r'Página \d+ de \d+\s*\n',                  # numeración de página
        r'^\d+\s*$',                                 # líneas que solo tienen números (nros de página)
        r'www\.udea\.edu\.co\s*\n',
        r'Medellín,?\s*\n',
    ]
    
    for patron in patrones_a_eliminar:
        texto = re.sub(patron, '', texto, flags=re.MULTILINE | re.IGNORECASE)
    
    return texto


def normalizar_puntuacion(texto: str) -> str:
    """Corrige problemas comunes de puntuación."""
    # Espacio antes de punto o coma (error de OCR frecuente)
    texto = re.sub(r'\s+([.,;:!?])', r'\1', texto)
    # Múltiples puntos seguidos normalizados
    texto = re.sub(r'\.{4,}', '...', texto)
    return texto


def eliminar_lineas_vacias_excesivas(texto: str) -> str:
    """Máximo 2 líneas vacías seguidas."""
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    return texto


def corregir_errores_ocr_espanol(texto: str) -> str:
    """
    Corrige confusiones típicas del OCR en textos en español.
    Estas son heurísticas: ajusta según los errores que veas en tus documentos.
    """
    correcciones = {
        r'\b0([a-záéíóúñ])\b': r'o\1',  # 0 confundido con 'o'
        r'\bl([0-9])': r'I\1',           # l confundida con I antes de número
        r'articu1o': 'artículo',          # error frecuente en reglamentos
        r'parágra[f]o': 'parágrafo',
        r'Articu1o': 'Artículo',
        r'rn([aeiou])': r'm\1',           # 'rn' confundido con 'm'
    }
    
    for patron, reemplazo in correcciones.items():
        texto = re.sub(patron, reemplazo, texto, flags=re.IGNORECASE)
    
    return texto
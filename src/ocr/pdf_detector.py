import fitz  # pymupdf

def calcular_densidad_texto(pdf_path: str, paginas_muestra: int = 3) -> float:
    """
    Retorna la densidad promedio de texto por página (caracteres / área).
    - Valor alto  → PDF nativo con texto seleccionable
    - Valor bajo  → PDF escaneado (imagen), necesita OCR
    """
    doc = fitz.open(pdf_path)
    densidades = []
    
    # Revisa solo las primeras N páginas para ser rápido
    paginas_a_revisar = min(paginas_muestra, len(doc))
    
    for num_pag in range(paginas_a_revisar):
        pagina = doc[num_pag]
        texto = pagina.get_text("text")
        area = pagina.rect.width * pagina.rect.height
        
        # Densidad = caracteres por unidad de área
        densidad = len(texto.strip()) / area if area > 0 else 0
        densidades.append(densidad)
    
    doc.close()
    return sum(densidades) / len(densidades) if densidades else 0.0


def es_pdf_escaneado(pdf_path: str, umbral: float = 0.001) -> bool:
    """
    Retorna True si el PDF necesita OCR (es una imagen escaneada).
    El umbral 0.001 es conservador: si hay duda, usa OCR.
    """
    densidad = calcular_densidad_texto(pdf_path)
    return densidad < umbral


def diagnosticar_pdf(pdf_path: str) -> dict:
    """Retorna un diagnóstico completo del PDF."""
    doc = fitz.open(pdf_path)
    densidad = calcular_densidad_texto(pdf_path)
    
    resultado = {
        "archivo": pdf_path,
        "paginas": len(doc),
        "densidad_texto": round(densidad, 6),
        "necesita_ocr": densidad < 0.001,
        "tipo": "escaneado" if densidad < 0.001 else "nativo"
    }
    doc.close()
    return resultado
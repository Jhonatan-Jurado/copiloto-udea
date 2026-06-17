"""
Script de prueba para verificar cada componente por separado.
Corre desde la raíz del proyecto:
    python src/ocr/test_ocr.py
"""
import sys
from pathlib import Path

# Permite importar los módulos vecinos
sys.path.insert(0, str(Path(__file__).parent))

PDF_PRUEBA = "data/raw/reglamento/documento_17917413.pdf"  # <-- cambia esto por el nombre real de tu PDF

def test_1_detector(pdf_path: str):
    """Prueba 1: ¿El detector identifica bien el tipo de PDF?"""
    print("\n" + "="*50)
    print("PRUEBA 1 — Detector de tipo de PDF")
    print("="*50)
    
    from pdf_detector import diagnosticar_pdf
    resultado = diagnosticar_pdf(pdf_path)
    
    print(f"  Archivo:       {resultado['archivo']}")
    print(f"  Páginas:       {resultado['paginas']}")
    print(f"  Densidad:      {resultado['densidad_texto']}")
    print(f"  Tipo:          {resultado['tipo'].upper()}")
    print(f"  Necesita OCR:  {resultado['necesita_ocr']}")
    
    return resultado


def test_2_extractor_nativo(pdf_path: str):
    """Prueba 2: Extrae texto de un PDF nativo."""
    print("\n" + "="*50)
    print("PRUEBA 2 — Extractor nativo (sin OCR)")
    print("="*50)
    
    from extractor_nativo import extraer_texto_nativo
    paginas = extraer_texto_nativo(pdf_path)
    
    print(f"  Páginas extraídas: {len(paginas)}")
    print(f"\n  --- Primeros 500 caracteres de la página 1 ---")
    primer_texto = paginas[0]["texto"][:500] if paginas else "(vacío)"
    print(primer_texto)
    
    return paginas


def test_3_extractor_ocr(pdf_path: str):
    """Prueba 3: Extrae texto usando OCR."""
    print("\n" + "="*50)
    print("PRUEBA 3 — Extractor OCR (solo primera página)")
    print("="*50)
    print("  (Puede tardar 10-30 segundos...)")
    
    from extractor_ocr import extraer_texto_ocr
    # Solo procesa la primera página para la prueba (más rápido)
    paginas = extraer_texto_ocr(pdf_path, dpi=200)
    paginas = paginas[:1]  # solo la primera
    
    print(f"  Páginas procesadas: {len(paginas)}")
    print(f"\n  --- Primeros 500 caracteres ---")
    primer_texto = paginas[0]["texto"][:500] if paginas else "(vacío)"
    print(primer_texto)
    
    return paginas


if __name__ == "__main__":
    # Verifica que el PDF existe
    if not Path(PDF_PRUEBA).exists():
        print(f"\n ERROR: No se encontró el archivo: {PDF_PRUEBA}")
        print(f" Pon un PDF en data/raw/ y actualiza la variable PDF_PRUEBA en este script")
        sys.exit(1)
    
    print(f"\nArchivo de prueba: {PDF_PRUEBA}")
    
    # Prueba 1: siempre corre
    diagnostico = test_1_detector(PDF_PRUEBA)
    
    # Prueba 2: extracción nativa (siempre corre, aunque sea escaneado)
    test_2_extractor_nativo(PDF_PRUEBA)
    
    # Prueba 3: OCR (pregunta antes porque puede tardar)
    respuesta = input("\n¿Correr también la prueba de OCR? (s/n): ")
    if respuesta.lower() == "s":
        test_3_extractor_ocr(PDF_PRUEBA)
    
    print("\n" + "="*50)
    print("Pruebas completadas")
    print("="*50)
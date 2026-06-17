import os
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from pdf_detector import diagnosticar_pdf, es_pdf_escaneado
from extractor_nativo import extraer_texto_nativo
from extractor_ocr import extraer_texto_ocr
from limpiador import limpiar_texto

# Carpetas de entrada y salida
CARPETA_ENTRADA = Path("preprocessing/data/raw")
CARPETA_SALIDA = Path("preprocessing/data/processed")

def procesar_pdf(pdf_path: Path) -> dict:
    """
    Procesa un PDF completo: detecta tipo → extrae texto → limpia → guarda.
    Retorna un reporte del procesamiento.
    """
    reporte = {
        "archivo": str(pdf_path),
        "nombre": pdf_path.name,
        "timestamp": datetime.now().isoformat(),
        "paginas": 0,
        "metodo": "",
        "caracteres_extraidos": 0,
        "caracteres_despues_limpieza": 0,
        "archivo_salida": "",
        "error": None
    }
    
    try:
        # 1. Detectar tipo de PDF
        diagnostico = diagnosticar_pdf(str(pdf_path))
        reporte["paginas"] = diagnostico["paginas"]
        reporte["metodo"] = diagnostico["tipo"]
        reporte["densidad_texto"] = diagnostico["densidad_texto"]
        
        print(f"\n  Tipo: {diagnostico['tipo'].upper()} "
              f"({diagnostico['paginas']} páginas)")
        
        # 2. Extraer texto según el tipo
        if diagnostico["necesita_ocr"]:
            print(f"  Aplicando OCR (puede tardar ~{diagnostico['paginas'] * 3}s)...")
            paginas = extraer_texto_ocr(str(pdf_path))
        else:
            print(f"  Extrayendo texto nativo...")
            paginas = extraer_texto_nativo(str(pdf_path))
        
        # 3. Unir todas las páginas con separador
        texto_completo = "\n\n---PÁGINA {}---\n\n".join(
            [p["texto"] for p in paginas]
        )
        # (forma más legible de hacerlo)
        partes = []
        for p in paginas:
            partes.append(f"--- PÁGINA {p['pagina']} ---\n{p['texto']}")
        texto_completo = "\n\n".join(partes)
        
        reporte["caracteres_extraidos"] = len(texto_completo)
        
        # 4. Limpiar el texto
        metodo_limpieza = "ocr" if diagnostico["necesita_ocr"] else "nativo"
        texto_limpio = limpiar_texto(texto_completo, fuente=metodo_limpieza)
        reporte["caracteres_despues_limpieza"] = len(texto_limpio)
        
        porcentaje = round((1 - len(texto_limpio)/len(texto_completo)) * 100, 1) if texto_completo else 0
        print(f"  Limpieza: {reporte['caracteres_extraidos']:,} → "
              f"{reporte['caracteres_despues_limpieza']:,} chars "
              f"({porcentaje}% eliminado)")
        
        # 5. Guardar el resultado manteniendo la estructura de carpetas
        ruta_relativa = pdf_path.relative_to(CARPETA_ENTRADA)
        ruta_salida = CARPETA_SALIDA / ruta_relativa.with_suffix(".txt")
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        
        # Cabecera con metadatos del documento
        cabecera = (
            f"FUENTE: {pdf_path.name}\n"
            f"RUTA_ORIGINAL: {pdf_path}\n"
            f"METODO_EXTRACCION: {diagnostico['tipo']}\n"
            f"PAGINAS: {diagnostico['paginas']}\n"
            f"PROCESADO: {reporte['timestamp']}\n"
            f"{'='*60}\n\n"
        )
        
        with open(ruta_salida, "w", encoding="utf-8") as f:
            f.write(cabecera + texto_limpio)
        
        reporte["archivo_salida"] = str(ruta_salida)
        print(f"  ✓ Guardado en: {ruta_salida}")
        
    except Exception as e:
        reporte["error"] = str(e)
        print(f"  ✗ ERROR: {e}")
    
    return reporte


def ejecutar_pipeline():
    """Procesa todos los PDFs en la carpeta de entrada."""
    
    # Busca todos los PDFs recursivamente
    pdfs = list(CARPETA_ENTRADA.rglob("*.pdf"))
    
    if not pdfs:
        print(f"No se encontraron PDFs en {CARPETA_ENTRADA}")
        return
    
    print(f"{'='*60}")
    print(f"PIPELINE OCR — UdeA Copiloto")
    print(f"{'='*60}")
    print(f"PDFs encontrados: {len(pdfs)}")
    print(f"Carpeta salida:   {CARPETA_SALIDA}")
    print(f"{'='*60}")
    
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    reportes = []
    
    for i, pdf_path in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] {pdf_path.name}")
        reporte = procesar_pdf(pdf_path)
        reportes.append(reporte)
    
    # Guarda el reporte general
    reporte_path = CARPETA_SALIDA / "pipeline_report.json"
    with open(reporte_path, "w", encoding="utf-8") as f:
        json.dump(reportes, f, ensure_ascii=False, indent=2)
    
    # Resumen final
    exitosos = [r for r in reportes if not r["error"]]
    con_error = [r for r in reportes if r["error"]]
    nativos = [r for r in exitosos if r["metodo"] == "nativo"]
    ocr = [r for r in exitosos if r["metodo"] == "escaneado"]
    
    print(f"\n{'='*60}")
    print(f"RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"Total procesados:  {len(reportes)}")
    print(f"  ✓ Exitosos:      {len(exitosos)} ({len(nativos)} nativos, {len(ocr)} OCR)")
    print(f"  ✗ Con error:     {len(con_error)}")
    print(f"Reporte guardado:  {reporte_path}")


if __name__ == "__main__":
    ejecutar_pipeline()
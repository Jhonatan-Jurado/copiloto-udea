import fitz  # pymupdf

def extraer_texto_nativo(pdf_path: str) -> list[dict]:
    """
    Extrae texto de un PDF nativo (con texto seleccionable).
    Retorna una lista de diccionarios, uno por página.
    """
    doc = fitz.open(pdf_path)
    paginas = []
    
    for num_pag in range(len(doc)):
        pagina = doc[num_pag]
        texto = pagina.get_text("text")  # extracción directa
        
        paginas.append({
            "pagina": num_pag + 1,
            "texto": texto,
            "metodo": "nativo"
        })
    
    doc.close()
    return paginas
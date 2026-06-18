// Scraper normativa.udea.edu.co - Node 18+ (sin dependencias)
// Ejecutar: node scraper.js

import { writeFileSync } from "node:fs";

const BASE = "https://normativa.udea.edu.co";

// Extrae verdocumento('codigo')
const RE_DOC = /verdocumento\(\s*'(\d+)'/g;
// Extrae verpagina('codigodocumento', 'codigoimagen', ...)
const RE_IMG = /verpagina\(\s*'(\d+)'\s*,\s*'(\d+)'/g;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Paso 1: paginar y obtener todos los codigodocumento
async function obtenerDocumentos() {
  const codigos = [];
  let page = 1;

  while (true) {
    const body =
      "tipobusqueda=indices&restringido=no&ordenarpor=indice2+DESC" +
      `&CurrentPage=${page}` +
      "&pdfini=0&pdffin=0&pdfcodigoinicial=0" +
      "&buscartodo=&tipodocumento=&dependencia=&asunto=&fecha=";

    const res = await fetch(`${BASE}/Documentos/Consultar`, {
      method: "POST",
      headers: { "content-type": "application/x-www-form-urlencoded" },
      body,
    });
    const html = await res.text();

    const enPagina = [...html.matchAll(RE_DOC)].map((m) => m[1]);
    if (enPagina.length === 0) break; // página sin coincidencias = fin

    codigos.push(...enPagina);
    console.log(`Página ${page}: ${enPagina.length} documentos`);
    page++;
    await sleep(500);
  }

  return [...new Set(codigos)]; // únicos
}

// Paso 2: para un documento, obtener sus codigoimagen
async function obtenerImagenes(codDoc) {
  const url = `${BASE}/Documentos/Ver?codigodocumento=${codDoc}&codigoimagen=&buscarpdf=`;
  const html = await (await fetch(url)).text();
  const imagenes = [...html.matchAll(RE_IMG)].map((m) => m[2]);
  return [...new Set(imagenes)];
}

// Principal
async function main() {
  console.log("== Obteniendo codigodocumento ==");
  const documentos = await obtenerDocumentos();
  console.log(`Total documentos: ${documentos.length}\n`);

  console.log("== Obteniendo codigoimagen y armando URLs ==");
  const urls = [];
  for (let i = 0; i < documentos.length; i++) {
    const codDoc = documentos[i];
    const imagenes = await obtenerImagenes(codDoc);
    for (const codImg of imagenes) {
      urls.push(
        `${BASE}/Documentos/Documento?codigodocumento=${codDoc}&codigoimagen=${codImg}&buscarpdf=`
      );
    }
    console.log(`[${i + 1}/${documentos.length}] ${codDoc}: ${imagenes.length} imágenes`);
    await sleep(300);
  }

  writeFileSync("urls.txt", urls.join("\n"), "utf-8");
  console.log(`\nListo: ${urls.length} URLs guardadas en urls.txt`);
}

main();
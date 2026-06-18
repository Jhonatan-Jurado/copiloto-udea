// Descarga los PDFs de urls.txt y los sube a POST /documents - Node 18+ (sin dependencias)
//
// Cada URL de urls.txt responde con un PDF descargable. Este script lo baja en
// memoria y lo reenvía al endpoint /documents como multipart/form-data, un
// archivo por request.
//
// El endpoint deriva `source` del nombre del archivo y RE-SUBIR el mismo nombre
// REEMPLAZA las filas previas. Como varias URLs comparten codigodocumento
// (varias paginas/imagenes por documento), el nombre se construye con
// codigodocumento + codigoimagen para que sea unico y no se pisen entre si.
//
// Uso:
//   node ocr.js
//   API_URL=http://localhost:8000/documents NIVEL=pregrado node ocr.js
//
// Variables de entorno:
//   API_URL      endpoint destino           (default http://localhost:8000/documents)
//   URLS_FILE    archivo con las URLs        (default urls.txt)
//   NIVEL        pregrado | posgrado         (default: se omite -> NULL)
//   CONCURRENCY  uploads en paralelo         (default 4)
//   MAX_MB       limite por archivo en MB    (default 25, igual que el endpoint)
//   RETRIES      reintentos de red por URL   (default 2)
//   STATE_FILE   log de URLs ya subidas      (default .uploaded.txt) para reanudar

import { readFileSync, appendFileSync, existsSync } from "node:fs";

const API_URL = process.env.API_URL || "http://localhost:8000/documents";
const URLS_FILE = process.env.URLS_FILE || "urls.txt";
const NIVEL = process.env.NIVEL || "";
const CONCURRENCY = Number(process.env.CONCURRENCY || 4);
const MAX_BYTES = Number(process.env.MAX_MB || 25) * 1024 * 1024;
const RETRIES = Number(process.env.RETRIES || 2);
const STATE_FILE = process.env.STATE_FILE || ".uploaded.txt";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// documento_3053415_3053416.pdf  ->  source "Documento 3053415 3053416"
function nombreArchivo(url) {
  const u = new URL(url);
  const doc = u.searchParams.get("codigodocumento") || "sin_doc";
  const img = u.searchParams.get("codigoimagen") || "sin_img";
  return `documento_${doc}_${img}.pdf`;
}

async function descargar(url) {
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok) throw new Error(`HTTP ${res.status} al descargar`);
  const buf = Buffer.from(await res.arrayBuffer());
  // El servidor responde application/octet-stream; validamos por los magic bytes.
  if (buf.length === 0) throw new Error("descarga vacia");
  if (buf.subarray(0, 5).toString("latin1") !== "%PDF-") {
    throw new Error("la respuesta no es un PDF (%PDF- ausente)");
  }
  if (buf.length > MAX_BYTES) {
    throw new Error(`${(buf.length / 1024 / 1024).toFixed(1)} MB > limite`);
  }
  return buf;
}

async function subir(buf, filename) {
  const form = new FormData();
  form.append("files", new Blob([buf], { type: "application/pdf" }), filename);
  if (NIVEL) form.append("nivel", NIVEL);

  const res = await fetch(API_URL, { method: "POST", body: form });
  const txt = await res.text();
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${txt.slice(0, 300)}`);

  let data;
  try {
    data = JSON.parse(txt);
  } catch {
    throw new Error(`respuesta no-JSON: ${txt.slice(0, 200)}`);
  }
  return data.results?.[0] || data;
}

async function procesar(url) {
  const filename = nombreArchivo(url);
  let ultimoError;
  for (let intento = 0; intento <= RETRIES; intento++) {
    try {
      const buf = await descargar(url);
      const r = await subir(buf, filename);
      return { url, filename, ...r };
    } catch (e) {
      ultimoError = e;
      if (intento < RETRIES) await sleep(500 * (intento + 1));
    }
  }
  return { url, filename, status: "error", error: ultimoError?.message };
}

async function main() {
  if (NIVEL && NIVEL !== "pregrado" && NIVEL !== "posgrado") {
    console.error(`NIVEL invalido: "${NIVEL}" (use pregrado | posgrado)`);
    process.exit(1);
  }

  const todas = readFileSync(URLS_FILE, "utf-8")
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.startsWith("http"));

  // Reanudar: saltar las URLs ya subidas correctamente en una corrida previa.
  const hechas = existsSync(STATE_FILE)
    ? new Set(readFileSync(STATE_FILE, "utf-8").split("\n").filter(Boolean))
    : new Set();
  const pendientes = todas.filter((u) => !hechas.has(u));

  console.log(`Endpoint : ${API_URL}`);
  console.log(`Nivel    : ${NIVEL || "(omitido -> NULL)"}`);
  console.log(`URLs     : ${todas.length} total, ${hechas.size} ya subidas, ${pendientes.length} pendientes`);
  console.log(`Paralelo : ${CONCURRENCY}\n`);

  const resumen = { ok: 0, rejected: 0, empty: 0, ocr_error: 0, error: 0 };
  let chunks = 0;
  let i = 0;

  async function worker() {
    while (i < pendientes.length) {
      const idx = i++;
      const url = pendientes[idx];
      const r = await procesar(url);

      resumen[r.status] = (resumen[r.status] ?? 0) + 1;
      if (r.status === "ok") {
        chunks += r.chunks_inserted || 0;
        appendFileSync(STATE_FILE, url + "\n"); // marcar para reanudar
      }

      const etiqueta = `[${idx + 1}/${pendientes.length}]`;
      const detalle =
        r.status === "ok"
          ? `${r.source} (${r.chunks_inserted} chunks)`
          : `${r.error || r.status}`;
      console.log(`${etiqueta} ${r.status.toUpperCase().padEnd(9)} ${r.filename}  ${detalle}`);
    }
  }

  await Promise.all(Array.from({ length: CONCURRENCY }, worker));

  console.log("\n== Resumen ==");
  console.log(`ok=${resumen.ok}  rejected=${resumen.rejected}  empty=${resumen.empty}  ocr_error=${resumen.ocr_error}  error=${resumen.error}`);
  console.log(`chunks insertados: ${chunks}`);
}

main().catch((e) => {
  console.error("Fallo fatal:", e);
  process.exit(1);
});

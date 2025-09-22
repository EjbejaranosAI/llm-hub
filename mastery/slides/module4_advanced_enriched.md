---
marp: true
paginate: true
title: "Módulo 4 — Técnicas Avanzadas y Casos de Uso"
description: "Razonamiento (CoT, Self‑Consistency, Tree‑of‑Thoughts), Agentes (LangChain, HuggingGPT), RAG avanzado (híbrido, multi‑query, evaluación)."
class: lead
---

# Técnicas Avanzadas y Casos de Uso (30 min)

**Objetivo:** estructurar **razonamiento**, orquestar **herramientas** y construir **RAG** confiable con métricas claras.

^ **Notas:** Dividir en tres bloques: Razonamiento, Agentes, RAG.

---

## Chain‑of‑Thought (CoT)

**CoT** pide al modelo “pensar paso a paso” con ejemplos de cadenas de razonamiento. Esto mejora tareas aritméticas y de *commonsense* al **hacer explícitos** los pasos intermedios. Es más efectivo en **modelos grandes** o con *few‑shot* bien diseñado, aunque añade **coste de tokens** y no siempre supera alternativas más ligeras.

^ **Notas:** Aclara que es técnica de prompting, no cambia pesos.

---

## Self‑Consistency

En lugar de una única cadena, **muestrea varias** cadenas de pensamiento y elige la **respuesta más consistente**. Esta votación reduce sesgos de una única trayectoria y aumenta la robustez, a cambio de **más cómputo**.

^ **Notas:** Útil cuando existen múltiples caminos válidos hacia la solución (p. ej., problemas lógicos).

---

## Tree‑of‑Thoughts (ToT)

**ToT** generaliza CoT a una **búsqueda en árbol** de “pensamientos”: se generan y evalúan ramas con retrocesos si es necesario. Es útil para planificación, *puzzles* y código, donde conviene **explorar** y **evaluar** planes parciales antes de continuar.

^ **Notas:** Mencionar *beam search* y evaluadores heurísticos simples.

---

## Agentes LLM: herramientas y *routing*

Los **agentes** convierten al LLM en un **controlador** que decide **qué herramienta** usar (API, base de datos, código). Con **LangChain** puedes definir *tool/function calling* y **salidas estructuradas**; **HuggingGPT** muestra el patrón “LLM orquestador de modelos expertos”. Buenas prácticas: *timeouts*, *retries*, *audit log* y herramientas **idempotentes**.

^ **Notas:** Señalar que el *grounding* con RAG reduce alucinaciones.

---

## RAG avanzado — Recuperación

- **Híbrido**: combina BM25 (sparse) con embeddings densos (FAISS) y **re‑ranking** *cross‑encoder*.  
- **Multi‑Query**: genera variantes de la consulta para cubrir sinónimos y parafraseo.  
- **Chunking** con *overlap* y **metadatos** (título/fecha/sección) mejora *recall* y filtrado.

^ **Notas:** Recomendar *Reciprocal Rank Fusion* para fusionar listas de recuperadores.

---

## RAG avanzado — Evaluación

Mide la **recuperación** (Recall@k, **MRR**, NDCG) y la **respuesta** (*faithfulness*, relevancia). Construye un **golden set** pequeño y haz **revisión humana** periódica. No confíes solo en *perplexity* o *BLEU*; necesitas métricas **de tu tarea** y *guardrails* (citas, filtros).

^ **Notas:** Mostrar una matriz de confusión simple: documentos correctos recuperados vs. no recuperados.

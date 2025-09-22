---
marp: true
paginate: true
title: "Módulo 3 — Despliegue de LLMs en Producción"
description: "Latencia, coste y escalado. vLLM (PagedAttention), TGI, Ray Serve. Demo: vLLM cuantizado."
class: lead
---

# Despliegue de LLMs en Producción (20 min)

**Objetivo:** construir *serving* con **baja latencia**, **alto throughput** y **coste predecible** usando motores modernos y cuantización.

^ **Notas:** Orientar a patrones generales que funcionen con diferentes modelos.

---

## Retos reales y objetivos

La inferencia de LLMs está dominada por la **KV‑cache** y el **ancho de banda de memoria**. Para atender tráfico real, debemos maximizar el **continuous batching**, reutilizar contexto con **caching** y distribuir carga entre GPUs/replicas. Todo ello sin degradar la *UX* (p50/p95 de latencia).

^ **Notas:** Introducir métricas: TPS, TTFT (time‑to‑first‑token), *tail latency* y coste por 1k tokens.

---

## vLLM: PagedAttention + batching continuo

**vLLM** introduce **PagedAttention**, que **pagina** la KV‑cache en bloques no contiguos (como memoria virtual), reduciendo **fragmentación** y permitiendo **batching** agresivo con latencia estable. Además expone una **API OpenAI‑compatible**, integra cuantización (p.ej., **W8A8**) y se adapta a contextos largos.

^ **Notas:** Enfatizar que el beneficio crece con secuencias largas y *decoding* complejo.

---

## TGI (Hugging Face)

**Text Generation Inference (TGI)** es un motor de producción escrito en **Rust+Python** con **tensor parallel**, *prefix caching*, métricas Prometheus y *streaming*. Ofrece integración nativa con el *ecosistema HF* y soporte multi‑backend (TensorRT‑LLM, Gaudi, Neuron, etc.).

^ **Notas:** Útil si ya usas HF Endpoints o quieres observabilidad lista.

---

## Ray Serve para orquestación

**Ray Serve** facilita desplegar **múltiples endpoints**, *autoscaling* y *rolling updates* en clúster. Permite componer **pipelines multi‑modelo** (p.ej., RAG + re‑ranker + generación) con control fino de **réplicas** y recursos por deployment.

^ **Notas:** Mostrar YAML de autoscaling basado en cola de peticiones.

---

## Patrones de rendimiento (receta)

- **Batching** dinámico + **streaming** de tokens.  
- **Cuantización** (INT8/4, W8A8) reduce VRAM y mejora TPS.  
- **Prefix/KV caching** para plantillas y prompts comunes.  
- **Rate limiting** y colas para picos.

^ **Notas:** Medir por workload (long/short prompts) y mezclar SLO por cola.

---

## Demo — vLLM cuantizado (10 min)

1) Levantar servidor OpenAI‑compatible de vLLM.  
2) Cargar **checkpoint cuantizado** (GPTQ o W8A8 con calibración).  
3) Realizar *benchmark* básico (concurrencia N, prompts repetidos/variados), comparando contra `transformers` puro.

^ **Notas:** Reportar p50/p95/avg y memoria consumida. Mostrar cómo *tuneas* `max_num_batched_tokens`.

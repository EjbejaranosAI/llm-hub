---
marp: true
paginate: true
title: "Módulo 5 — Tendencias, Retos y Futuro (+ Extras)"
description: "MoE, modelos ligeros (TinyLLaMA, Mistral), multimodal (Gemini, LLaVA‑Next), model collapse, transparencia. Extras: BitNet b1.58 y cuantización trending."
class: lead
---

# Tendencias, Retos y Futuro (15 min)

**Objetivo:** entender hacia dónde evoluciona el stack (eficiencia, multimodalidad, datos) y riesgos como *model collapse*.

^ **Notas:** Cerrar el curso con visión y criterios de adopción.

---

## Mixture of Experts (MoE)

MoE divide la **capacidad** entre **expertos** y enruta cada *token* a un subconjunto, logrando **más capacidad efectiva** sin pagar el coste denso por token. Requiere **balanceo** y regularización del enrutamiento para evitar especialistas ociosos y *instabilidad* de entrenamiento.

^ **Notas:** Relacionar con Switch Transformers como ejemplo histórico.

---

## Modelos ligeros y eficientes

Modelos como **TinyLLaMA (1.1B)** y **Mistral 7B** demuestran que una **ingeniería de atención** (GQA, SWA) y entrenamiento cuidado permiten alto rendimiento con **coste de inferencia** reducido. Combinados con **cuantización**, habilitan *edge/on‑prem* y especialización por dominio.

^ **Notas:** Mostrar casos de uso en dispositivos limitados.

---

## Multimodalidad

Sistemas como **Gemini** y **LLaVA‑Next** integran texto‑imagen‑audio/video, habilitando análisis de documentos, visión industrial y asistentes más ricos. El reto pasa por **datos** y **alineamiento** multimodal, además de costes de entrenamiento/inferencia.

^ **Notas:** Conectar con agentes que llaman a modelos expertos de visión.

---

## Riesgos: *Model Collapse* y datos sintéticos

Entrenar repetidamente con **datos sintéticos** no curados puede provocar **pérdida de diversidad** y degradación a lo largo de generaciones de modelos. Mitigación: filtrar/etiquetar lo sintético, **watermarking**, y mezclar datos humanos curados.

^ **Notas:** Señalar valor estratégico de datasets humanos “pre‑AI”.

---

## Transparencia y seguridad con cuantización

La **cuantización** (INT8/4, W8A8) cambia distribuciones internas; audita su impacto en **sesgos** y **toxicity**. Publica una **card de despliegue** con hardware, formato, métricas y *guardrails*; monitoriza *drift* y añade *feedback loops*.

^ **Notas:** Recomendar pruebas A/B controladas por segmento de usuarios.

---

## Extras — Quantización en tendencia (BitNet + más)

- **BitNet b1.58**: pesos **ternarios** {‑1,0,1} (~1.58 bits) con rendimiento cercano a FP16; prometedor para **latencia/energía** y futuros **kernels/hardware** específicos.  
- Otras líneas: **FP8/FP4** en activaciones, **KV‑cache 3‑bit**, *learned rounding* (LSQ/AdaRound).

^ **Notas:** Esta sección es tuya para profundizar en BitNet con tu material.

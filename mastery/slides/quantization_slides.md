---
marp: true
paginate: true
theme: default
class: lead
title: "Quantización y Optimización de Inferencia (LLMs)"
description: "Fundamentos + métodos modernos (PTQ, QAT, Dynamic), LLM.int8, GPTQ, AWQ, SmoothQuant, KV-Cache, vLLM W8A8, GGUF."
---

# Quantización y Optimización de Inferencia (LLMs)

**Objetivo (~20 min):** reducir memoria/latencia • mantener calidad • elegir el método correcto  
**Qué veremos:** PTQ, QAT, **Dynamic**, LLM.int8, GPTQ, AWQ, SmoothQuant, KV-Cache, vLLM W8A8, GGUF

^ **Discurso (30–40s):** Vamos a ver cómo la cuantización permite ejecutar LLMs más rápido y barato sin perder mucha calidad. Cubrimos desde la base teórica hasta métodos modernos específicos para LLMs y un plan de demo.

---

## ¿Por qué cuantizar LLMs?

- Memoria y ancho de banda ↓ → **latencia ↓ / throughput ↑**
- Despliegue en **GPU/CPU modestas** y contextos más largos
- Pérdida de calidad **controlada** si se elige bien el método

^ **Discurso (40s):** La inferencia en LLMs está limitada por memoria y transferencias. Cuantizar baja ambos y acelera la ejecución. El reto es hacerlo sin dañar mucho la calidad.

---

## Operador de cuantización (uniforme affine)

- De **FP32/FP16 → INT8/INT4**
- Fórmula: \(x_q = clip(round(x/s) + z)\),  \(x \approx s(x_q - z)\)
- Parámetros: **escala** (s) y **zero-point** (z)
- Dónde aplico s,z: **per-tensor / per-canal / group-wise (bloques)**

^ **Discurso (60s):** La forma más usada en LLMs es la cuantización uniforme affine. Convertimos floats a enteros con una escala y un punto cero. Elegir la granularidad impacta directamente en el error.

---

## Granularidad y redondeo

- **Per-tensor** (simple) vs **per-canal / group-wise** (mejor precisión)
- Redondeo: **RTN** (baseline), **AdaRound** (redondeo aprendido), **LSQ** (aprende el paso)
- A **4 bits**: granularidad fina + buen redondeo marcan la diferencia

^ **Discurso (60s):** A 8 bits casi todo funciona. A 4 bits el detalle importa: usar group-wise o per-canal y técnicas como AdaRound/LSQ reduce la pérdida.

---

## Mapa mental: PTQ, QAT y **Dynamic**

- **PTQ**: cuantizas **después** de entrenar → rápido, sin reentrenar
- **QAT**: simulas INT-k **durante** el entrenamiento → más fiel, más caro
- **Dynamic Quantization (DQ)**: pesos INT8; **activaciones “al vuelo”** (Linear/LSTM)

^ **Discurso (70s):** Empieza por PTQ; si la calidad cae, pasa a QAT. En medio, Dynamic: PyTorch cuantiza pesos y cuantiza activaciones en tiempo de inferencia, útil cuando no puedes reentrenar y quieres resultados rápidos.

---

## LLM.int8(): outliers y mezcla 8/16-bit

- Detecta **outliers** y los computa en **16-bit**; el resto en **8-bit**
- Mantiene calidad con ~50% de memoria vs FP16
- Disponible en **bitsandbytes** / Transformers

^ **Discurso (50s):** LLM.int8 es muy práctico: trata outliers en 16-bit y deja el resto en 8-bit. Suele mantener la calidad y ahorrar memoria de forma notable.

---

## PTQ “peso-solo”: GPTQ y AWQ

- **GPTQ**: PTQ *one-shot* (info de 2º orden), 3–4 bits **weight-only**
- **AWQ**: protege ~**1%** de pesos “salientes” guiado por **activaciones**
- Checkpoints W4/W3 listos para servir

^ **Discurso (60s):** Para LLMs, GPTQ es el estándar en 3–4 bits peso-solo. AWQ observa activaciones para decidir qué pesos proteger. Permiten grandes ahorros con poco esfuerzo.

---

## PTQ peso+activación (W8A8): SmoothQuant

- Problema: **outliers** en activaciones rompen INT8
- **SmoothQuant** “traslada” outliers de activaciones → **pesos**
- Habilita **W8A8** con mínima pérdida (training-free)

^ **Discurso (45s):** Si además de pesos quieres activaciones en INT8, SmoothQuant suaviza activaciones para que el hardware INT8 sea estable.

---

## **Dynamic Quantization** (práctica)

- Aplica a un modelo ya entrenado (**no** reentrena)
- PyTorch: `quantize_dynamic(model, {nn.Linear, nn.LSTM}, dtype=qint8)`
- Ganancia típica en **CPU**; útil sin dataset de calibración

^ **Discurso (60s):** Dynamic es literalmente un switch: conviertes capas Linear y LSTM y ejecutas activaciones cuantizadas al vuelo. Ideal para CPU y entornos donde no hay datos de calibración.

---

## Más métodos útiles

- **ZeroQuant / ZeroQuant-FP / HERO**: PTQ end-to-end, W8A8 y FP8/FP4
- **RPTQ**: reordena canales para **A3** (activaciones 3-bit)
- **BRECQ**: reconstrucción por bloque (mejora PTQ a muy bajos bits)
- **ACIQ**: *clipping* analítico de activaciones (elige rangos óptimos)

^ **Discurso (60s):** El ecosistema se mueve rápido. Estas técnicas empujan bajo bit con menos pérdida: desde reordenar canales (RPTQ) hasta clipping analítico (ACIQ) o reconstrucción por bloque (BRECQ).

---

## **KV-cache quantization** (contextos largos)

- En long context, la **KV cache** domina la memoria
- **KVQuant**: pre-RoPE, per-canal, no uniforme, “dense+outliers”
- Hasta **3-bit** en K/V con <0.1 ppl extra; contextos **1–10M**

^ **Discurso (60s):** Con contextos largos la memoria se la lleva la KV-cache. KVQuant muestra que se puede bajar a 3 bits con poca pérdida y habilitar contextos enormes.

---

## Herramientas y formatos

- **bitsandbytes** (LLM.int8, 4/8-bit), **GPTQ** toolchains
- **vLLM W8A8** (requiere **calibración** de activaciones)
- **GGUF/GGML + llama.cpp** (CPU/edge): Q4/Q5/Q8; conversión desde HF

^ **Discurso (50s):** En GPU: bitsandbytes y GPTQ son lo más directo; para W8A8, vLLM soporta activaciones INT8/FP8 calibradas. En CPU, GGUF/llama.cpp es el estándar.

---

## Trade-offs y buenas prácticas

- **INT8** suele ser “seguro”; **INT4** ahorra más pero exige cuidado
- **Weight-only** (W4/W3) vs **W8A8** (peso+activación)
- Proteger **QKV/outliers**; medir **VRAM/latencia/calidad** en tu tarea

^ **Discurso (45s):** Receta pragmática: empieza por INT8 peso-solo, mide; si necesitas más ahorro, intenta INT4 con técnicas outlier-aware o W8A8 bien calibrado. Decide con métricas de tu caso.

---

## Mini-demo (plan)

- Modelo 7B (LLaMA/Mistral) **FP16 → medir** VRAM/latencia
- Cargar **LLM.int8()** → repetir métricas
- (Opcional) checkpoint **GPTQ W4** y comparar

^ **Discurso (40s):** Mostramos el “antes/después”: memoria y tiempo por prompt. Suele verse ~2× menos memoria y mejor latencia, con mínima caída de calidad.

---

## Referencias clave (enlazadas en el README del módulo)

- **LLM.int8** (Dettmers et al., NeurIPS 2022)
- **GPTQ** (Frantar et al., ICLR 2023)
- **AWQ** (Lin et al., 2023)
- **SmoothQuant** (Xiao et al., 2023)
- **Dynamic Quantization** (Docs/Tutorial PyTorch)
- **ZeroQuant-Series / ZeroQuant-FP**
- **RPTQ** (Yuan et al., 2023)
- **ACIQ** (Banner et al., 2018) • **BRECQ** (Nagel et al., 2020)
- **KVQuant** (Hooper et al., NeurIPS 2024)
- **vLLM W8A8** docs • **GGUF/llama.cpp** docs

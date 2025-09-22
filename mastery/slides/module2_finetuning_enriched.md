---
marp: true
paginate: true
title: "Módulo 2 — Optimización de Entrenamiento y Fine‑tuning (LLMs)"
description: "QLoRA, LoRA/AdaLoRA/IA3, Mixture of Adapters, FSDP/DeepSpeed/Megatron‑LM, Gradient Checkpointing y ZeRO. Demo con QLoRA."
class: lead
---

# Optimización de Entrenamiento y Fine‑tuning (20 min)

**Objetivo:** entender *por qué* PEFT funciona y *cómo* combinarlo con entrenamiento distribuido para afinar LLMs con poca VRAM sin perder calidad.  
**Ítems:** QLoRA • LoRA/AdaLoRA/IA3 • Mixture of Adapters • FSDP • DeepSpeed/ZeRO • Megatron‑LM • Gradient Checkpointing • **Demo QLoRA**

^ **Notas:** Introducir la idea: mover el aprendizaje a unos pocos parámetros (adapters/LoRA) y usar técnicas de distribución/ahorro de memoria para escalar.

---

## PEFT en contexto (¿por qué no full fine‑tune?)

El *full fine‑tuning* actualiza **todos** los pesos, lo que exige mucha VRAM y almacenamiento de checkpoints. **PEFT** (Parameter‑Efficient Fine‑Tuning) congela el modelo y entrena **pequeños módulos adicionales** (adapters, low‑rank, escaladores), logrando precisión comparable con **miles de veces menos parámetros entrenables** y **3× menos memoria** en casos reportados por LoRA.  

^ **Notas:** Resaltar que PEFT reduce costo y acelera iteración; citaremos LoRA como base y QLoRA como evolución práctica.

---

## LoRA (Low‑Rank Adaptation) — la base

LoRA inserta matrices de **bajo rango (A·Bᵀ)** en capas lineales del Transformer y solo entrena esas matrices. La intuición: la actualización óptima suele tener **baja “rango intrínseco”**, por lo que bastan pocos grados de libertad para adaptar el modelo. En práctica, LoRA logra calidad *on‑par* con *full fine‑tune* con **~10,000× menos parámetros entrenables** y **sin latencia extra en inferencia**.

^ **Notas:** Recomendar iniciar con `r=8–16`, `target_modules=["q_proj","v_proj"]` y ajustar según tarea.

---

## QLoRA — LoRA + modelo base 4‑bit (NF4)

**QLoRA** cuantiza el modelo base a **4‑bit NF4** (normal‑float) y entrena LoRA encima. Aporta **double quantization** (ahorra bits al cuantizador) y **paged optimizers** (evitan picos de memoria). Resultado: *fine‑tuning* de modelos hasta **65B en 48GB** de VRAM **manteniendo** calidad cercana a 16‑bit en benchmarks.

^ **Notas:** Señalar que NF4 modela bien distribuciones de pesos; los *paged optimizers* amortiguan memoria de estados del optimizador.

---

## AdaLoRA e IA3 — refinar el presupuesto

- **AdaLoRA** asigna **rango LoRA dinámico por capa** según importancia (medida con SVD/gradientes), usando mejor el presupuesto cuando **r es bajo**.  
- **IA3** no añade proyecciones sino **vectores que escalan activaciones internas** (inhibe/realza), reduciendo aún más parámetros y manteniendo estabilidad.

^ **Notas:** Pautas: probar IA3 cuando la VRAM es mínima; usar AdaLoRA si observas cuellos en capas específicas.

---

## Mixture of Adapters (MoA)

En lugar de un único adapter, **varios adapters especializados** se combinan mediante **enrutamiento estocástico** o un *router* liviano. Así puedes **multi‑tarea/multi‑dominio** sin *catastrophic forgetting*, y fusionar adapters (*merging*) para despliegue ligero.

^ **Notas:** Útil cuando tu organización tiene muchos verticales pequeños con datos propios.

---

## Entrenamiento distribuido: el mapa

- **FSDP (PyTorch):** *sharding* de parámetros/gradientes/estados → memoria lineal con el número de GPUs.  
- **DeepSpeed/ZeRO:** particiona estados del optimizador (**ZeRO‑1/2/3**), con opciones de **offload** a CPU/NVMe.  
- **Megatron‑LM:** **tensor/pipeline parallel** para escalar a decenas de GPUs manteniendo *throughput*.

^ **Notas:** Mezclar PEFT con **ZeRO‑2** y **gradient checkpointing** suele dar el mejor “bang for the buck”.

---

## Ahorro de memoria adicional

- **Gradient checkpointing:** guarda activaciones clave y **recomputa** el resto en *backward* → reduce picos de VRAM.  
- **ZeRO optimizer:** distribuye estados del optimizador y gradientes entre procesos → evita replicación.  
- **Precisión mixta:** bf16/fp16 donde sea estable.

^ **Notas:** Recordar ajustar `grad_accumulation_steps` para mantener *effective batch size*.

---

## Prácticas y *pitfalls*

- Ajusta **r** y **lora_alpha** con validación; empieza pequeño y sube.  
- **Target modules**: `q_proj`/`v_proj` suelen bastar; añadir `k_proj/o_proj` si falta capacidad.  
- Evita *overfitting* con **dropout LoRA** y *early stopping*.  
- Mide en la **tarea final** (no solo perplexity).

^ **Notas:** Llevar registro de *learning rate finders* y *sweeps* pequeños.

---

## Demo — QLoRA en Alpaca (10 min)

1) Carga checkpoint 7B **4‑bit** con `load_in_4bit=True`.  
2) Inserta LoRA (`r=16`, `alpha=16`, `dropout=0.05`).  
3) Entrena 1–2 épocas en **subset** (1k–3k ejemplos).  
4) Evalúa y guarda **adaptadores** para mezclar/servir.

^ **Notas:** Mostrar `model.print_trainable_parameters()` y una comparación de métricas simples antes/después.

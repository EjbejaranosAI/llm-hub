# 🔵 LLM Mastery – Curso Avanzado de Large Language Models

**Duración total:** ~120 minutos  
**Prerequisito:** Haber completado el curso *LLM Foundations*  

Curso previo: [LLM Foundations](../foundations/syllabus.md)  

> ⚠️ Nota: En desarrollo

---

## 🧩 Módulo 1 – Cuantización y Optimización de Inferencia
- ¿Qué es cuantización?  
- Técnicas:  
  - Post-training quantization (PTQ)  
  - Quantization-aware training (QAT)  
  - GPTQ, AWQ, SmoothQuant  
- Tradeoffs: velocidad ⚡, memoria 💾, precisión 🎯  
- Herramientas prácticas:  
  - `bitsandbytes`, AutoGPTQ, `llm.int8()` (Hugging Face)  
  - GGUF / GGML (CPU inference)  
- **Demo:** Cuantizar un modelo *LLaMA-2-7B* y comparar memoria/velocidad  

---

## 🛠️ Módulo 2  – Optimización de Entrenamiento y Fine-tuning
- QLoRA (LoRA + quantization en training)  
- AdaLoRA y IA3  
- Mixture of Adapters  
- Entrenamiento distribuido: FSDP, DeepSpeed, Megatron-LM  
- Técnicas de *gradient checkpointing* y ZeRO optimizer  
- **Demo:** Fine-tuning con QLoRA en un subset reducido de *Alpaca*  

---

## 🚀 Módulo 3  – Despliegue de LLMs en Producción
- Desafíos reales:  
  - Latencia (batching, caching)  
  - Costos (GPU sharing, quantized inference)  
  - Escalabilidad (multi-GPU, clusters)  
- Herramientas:  
  - **vLLM** (fast inference con paged attention)  
  - **TGI** (Text Generation Inference – Hugging Face)  
  - **Ray Serve** para escalamiento  
- **Demo:** Deploy de un modelo cuantizado en vLLM y comparación con *transformers* estándar  

---

## 🧠 Módulo 4  – Técnicas Avanzadas y Casos de Uso
- **Razonamiento Avanzado:**  
  - Chain-of-Thought optimizado (CoT prompting)  
  - Self-Consistency y Tree-of-Thoughts 🌳  
- **Agentes basados en LLMs:**  
  - LangChain + herramientas externas (API calls, ejecución de código)  
  - HuggingGPT: coordinación entre modelos especializados  
- **RAG Avanzado:**  
  - Indexado híbrido (denso + BM25)  
  - Multi-query retrievers  
  - Evaluación en RAG: *recall@k*, *MRR*  
- **Demo:** Construir un *LLM Agent* que use RAG + ejecución de código  

---

## 🔮 Módulo 5  – Tendencias, Retos y Futuro
- Modelos híbridos: LLMs + MoE (Mixture of Experts)  
- Modelos ligeros para *Edge AI*: TinyLLaMA, Mistral  
- Avances en multimodalidad: Gemini, LLaVA-Next  
- Riesgos: *model collapse* por entrenamiento recursivo  
- Transparencia en cuantización y sesgos ⚖️  

---

## ✅ Objetivos del curso
- Comprender y aplicar técnicas avanzadas de **optimización e inferencia**.  
- Dominar métodos de **fine-tuning eficiente** (PEFT, QLoRA, etc.).  
- Saber cómo **desplegar LLMs en producción** de forma escalable y rentable.  
- Explorar casos de uso avanzados: **RAG, agentes, razonamiento complejo**.  
- Conocer las **tendencias futuras** y los retos en investigación y producción.  

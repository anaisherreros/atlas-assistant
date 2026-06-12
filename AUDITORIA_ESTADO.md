# Auditoría de estado — atlas-assistant

**Fecha:** 2026-06-12  
**Alcance:** repositorio `atlas-assistant` + endpoints REST/MCP de `atlas-vital`  
**Método:** lectura estática del código. Donde se indica «requiere runtime», hace falta ejecutar el bot para confirmarlo.

---

## Resumen ejecutivo

**Estado en una frase:** El bot funciona como un wrapper de Telegram sobre Claude Sonnet 4.5 + MCP de Atlas Vital, con una ruta determinista paralela para ~12 frases exactas; el resto es conversación libre donde la confianza depende casi por completo de que el modelo invoque herramientas MCP y las interprete bien.

**Tedio (3 causas principales):**

1. **Doble cerebro desincronizado** — ruta rápida con sintaxis rígida (`deterministic_handlers.py`) vs. ruta lenta con confirmaciones en los prompts (`prompts/agents/personal.md:132`, `performance.md:104`). El lenguaje natural casi siempre cae en la segunda.
2. **Latencia acumulada** — cada mensaje no determinista = 1 llamada `beta.messages.create` con MCP (`claude.py:37-44`), sin Haiku ni router barato; además `get_dashboard()` hace 3 HTTP secuenciales cuando se usa (`atlas_client.py:37-45`).
3. **Ceremonia innecesaria** — mensaje de transición de agente (`router.py:5-10`, `conversation_flow.py:168-170`), menú de ayuda largo (`deterministic_handlers.py:722-747`), prompts que piden 3-4 párrafos (`personal.md:140-146`).

**Desconfianza (3 causas principales):**

1. **El contexto prometido no se inyecta** — los prompts dicen «recibirás contexto dinámico» pero `fetch_context_for_agent` (`agents/context.py:264`) **no se llama desde ningún flujo** (verificado por grep en todo el repo).
2. **URLs rotas en `atlas_client.py`** — 4 endpoints que no existen o difieren en Atlas Vital (tabla en Fase 3).
3. **Confirmaciones engañosas** — la ruta determinista dice «Tarea creada» sin leer `created: false` del API (`deterministic_handlers.py:855-862`); Claude puede responder en texto sin tool call real (`claude.py:53-63`); moneda hardcodeada a EUR en Suiza (`deterministic_handlers.py:205`).

---

## Fase 1 — Mapa del sistema

### Ciclo de vida de un mensaje

```
Telegram → main.py:44 handle_text
        → telegram_runtime.py:46 process_text_message
        → router.detect_agent (keywords)
        → try_handle_deterministic_message (REST vía atlas_client)
        → si no match: build_agent_system_prompt + generate_with_tools (MCP)
        → reply_text (chunked 4096)
```

| Paso | Archivo | Qué hace |
|------|---------|----------|
| 1 | `main.py:44` | Solo `MessageHandler(TEXT & ~COMMAND)` — **no hay handlers de `/comandos`** |
| 2 | `telegram_runtime.py:62` | `send_chat_action(TYPING)` |
| 3 | `conversation_flow.py:164-172` | Detecta/cambia agente; mensaje de transición si cambia |
| 4 | `conversation_flow.py:175-190` | Ruta determinista (REST vía `atlas_client`) |
| 5 | `conversation_flow.py:192-223` | Ruta agente: historial DB + Claude MCP |
| 6 | `database.py` | Persiste mensajes; memoria cada 20 msgs usuario |

### Agentes reales en código

| Clave | Nombre | Prompt | Activo |
|-------|--------|--------|--------|
| `personal` | Asistente personal | `prompts/agents/personal.md` | Sí (default) |
| `coach` | Coach | `prompts/agents/coach.md` | Sí |
| `performance` | Especialista rendimiento (nutri + entreno fusionados) | `prompts/agents/performance.md` | Sí |
| `financial` | Asesor financiero | `prompts/agents/financial.md` | Sí |
| `nutritionist` / `trainer` | Alias → `performance` | Reexportan `performance.AGENT` | Solo compatibilidad DB |

**No existen** agentes separados de nutricionista y entrenador (`agents/nutritionist.py`, `agents/trainer.py` son reexports de 2 líneas).

**Resumen de system prompts:**

| Agente | Enfoque | Reglas clave |
|--------|---------|--------------|
| personal | Mano derecha operativa, priorización, derivación | Confirma antes de escribir; máx. 3-4 párrafos |
| coach | Patrones mentales, ikigai, revisiones | Confrontación amorosa; confirma antes de escribir |
| performance | Nutrición antiinflamatoria, gym adaptado, lesiones | Confirma antes de registrar en Atlas |
| financial | Finanzas, crypto, inversión en marca personal | No inventar datos financieros |

### Router (no es LLM)

`router.py:13-82` — keywords en español, sin Claude:

- Frases explícitas: «pásame con coach/finanzas/…»
- Keywords nutrición o entreno → `performance`
- Keywords coach (meta, propósito, ikigai…) → `coach`
- Si no hay match → mantiene agente actual

**Riesgo:** decir «comida» o «gym» con `personal` activo cambia agente y muestra «Conectando con…» (`conversation_flow.py:168-170`).

### Mapa agente → capacidades → endpoints → modelo

| Agente | Capacidades | Endpoints / tools | Modelo |
|--------|-------------|-------------------|--------|
| personal | Coordinación; todas las tools MCP | MCP: todas (`atlas-vital/mcp_app/server.py`) | `claude-sonnet-4-5` |
| coach | Deseos, goals, revisiones, relaciones self | MCP: subset en `agent_tool_policy.py` (**no enforced** en runtime) | `claude-sonnet-4-5` |
| performance | Hábitos, salud, ejercicio | MCP: `log_habit_completion`, `log_health`, `log_exercise`, etc. | `claude-sonnet-4-5` |
| financial | Transacciones, patrimonio | MCP: `create_transaction`, `get_finance`, etc. | `claude-sonnet-4-5` |
| Router | Clasificación por keywords | N/A | Ninguno |
| Determinista | ~12 frases exactas | REST directo `atlas_client.py` | Ninguno |
| Memoria (cada 20 msgs) | Resumen conversacional | N/A | `claude-sonnet-4-5` |

**Haiku:** no existe en el código.

### Acciones de escritura completas

**Vía MCP** (ruta agente — `atlas-vital/mcp_app/server.py`):

| Acción | Tool MCP | Endpoint REST |
|--------|----------|---------------|
| Crear tarea | `create_task` | `POST /api/assistant/tasks/create/` |
| Actualizar tarea | `update_task` | `POST /api/assistant/tasks/update/` |
| Completar tarea | `complete_task` | `POST /api/assistant/tasks/complete/` |
| Eliminar tarea | `delete_task` | `POST /api/assistant/tasks/delete/` |
| Crear deseo | `create_desire` | `POST /api/assistant/desires/create/` |
| Actualizar / eliminar deseo | `update_desire` / `delete_desire` | `.../desires/update/` / `delete/` |
| Crear / actualizar / eliminar goal | `create_goal` etc. | `.../goals/create/` etc. |
| Crear / actualizar / eliminar hábito | `create_habit` etc. | `.../habits/create/` etc. |
| Marcar hábito | `log_habit_completion` | `POST /api/assistant/habits/log/` |
| Registrar salud | `log_health` | `POST /api/assistant/health/log/` |
| Actualizar salud | `update_health` | `POST /api/assistant/health/update/` |
| Registrar ejercicio | `log_exercise` | `POST /api/assistant/exercise/log/` |
| Crear transacción | `create_transaction` | `POST /api/assistant/finance/transaction/` |
| Eliminar transacción | `delete_transaction` | `POST /api/assistant/finance/transaction/delete/` |
| Snapshot patrimonio | `create_patrimony_snapshot` | `POST /api/assistant/finance/patrimony/` |
| Crear / actualizar relación | `create_relationship` / `update_relationship` | `.../relationships/create/` etc. |
| Log relación / self | `log_relationship` / `log_self_relationship` | `.../relationships/log/` / `self/log/` |
| Revisión diaria / semanal / mensual | `create_daily_review` etc. | `.../reviews/daily/create/` etc. |

**Vía REST directo** (ruta determinista + automatizaciones):

`create_task`, `create_habit`, `complete_task`, `log_habit`, `create_transaction` — solo con sintaxis exacta.

**`tools.py` + `dispatch_atlas_tool`:** código muerto; MCP lo reemplazó.

### Acciones de solo lectura

| Fuente | Cuándo | Qué inyecta |
|--------|--------|-------------|
| MCP tools (agente) | Claude decide | `get_today`, `get_dashboard`, etc. bajo demanda |
| `atlas_client` REST (determinista) | Match de frase | `get_today`, `get_tasks_pending`, `get_calendar`, `get_finance` |
| Memoria conversacional | Si existe summary en DB | Bloque texto en system prompt (`conversation_flow.py:198-200`) |
| Automatización diaria | 07:30 / 21:30 | `get_today`, `get_last_daily_review` (`daily_automation.py`) |

**Niveles `none` / `today` / `finance` / `full`:** implementados en `message_classification.py:classify_context` pero **sin referencias en el proyecto** — no conectados.

**`fetch_context_for_agent`:** implementado en `agents/context.py:264` pero **nunca se llama** — los prompts prometen contexto dinámico que no llega al runtime.

### Código muerto / features a medias

| Módulo | Estado |
|--------|--------|
| `message_classification.py` | Completo, sin referencias |
| `agents/context.py` | Completo, sin referencias |
| `agent_tool_policy.py` | Política de tools por agente, sin referencias |
| `tools.py` → `dispatch_atlas_tool` | Sin referencias |
| `agents/nutritionist.py`, `trainer.py` | Aliases muertos |
| `claude.py` params `tools`, `max_tool_loops` | Documentados como «ignorados» (`claude.py:30-31`) |
| Prompts «contexto dinámico» | Prometido en `.md`, no implementado |
| `.env.example` | Falta `ATLAS_VITAL_URL` y `ASSISTANT_API_KEY` que `main.py:21-26` exige |

---

## Fase 2 — Traza de casos cotidianos

### A. «marca hábito beber agua» (determinista)

| Métrica | Valor |
|---------|-------|
| Llamadas Claude | 0 |
| HTTP Atlas | 2 secuenciales: `GET /api/assistant/today/` → `POST /api/assistant/habits/log/` |
| Latencia estimada | ~0,5–2 s (sin LLM) |
| Turnos usuario | 1 si el hábito resuelve por título; 2 si hay ambigüedad (`deterministic_handlers.py:583-585`) |

### B. «registra gasto gasolina 45 hoy» (determinista)

| Métrica | Valor |
|---------|-------|
| Llamadas Claude | 0 |
| HTTP Atlas | 1: `POST /api/assistant/finance/transaction/` |
| Latencia estimada | ~0,3–1 s |
| Turnos usuario | 1 — confirmación concreta pero **sin categoría** y moneda **EUR** (`deterministic_handlers.py:205`) |

### C. «qué tengo hoy» (determinista)

| Métrica | Valor |
|---------|-------|
| Llamadas Claude | 0 |
| HTTP Atlas | 1: `GET /api/assistant/today/` |
| Latencia estimada | ~0,3–1 s |
| Turnos usuario | 1 — lista formateada con IDs |

### Contraste: «ya hice yoga» (lenguaje natural — más habitual)

| Métrica | Valor |
|---------|-------|
| Llamadas Claude | 1 `beta.messages.create` mínimo; **sospecha:** varios ciclos MCP internos (`claude.py:35`) |
| HTTP Atlas | 1+ vía MCP si el modelo coopera |
| Latencia estimada | 3–15 s típico — **requiere medición en runtime** |
| Turnos usuario | 2–4: prompt pide confirmar antes de registrar; router puede cambiar a performance; posible repregunta de cuál hábito |

---

## Fase 3 — Auditoría de confianza

### Tabla de fallos silenciosos

| Archivo | Línea | Qué falla | Qué ve el usuario |
|---------|-------|-----------|-------------------|
| `agents/context.py` | 393-395 | Cualquier error → `return "{}"` | Nada (módulo no conectado) |
| `agents/context.py` | 322-330 | `get_health_emotional` → 404 | Nada (no conectado) |
| `conversation_flow.py` | 44-46, 217-220 | Si `tools_used` y respuesta <100 palabras → no guarda historial | Amnesia en el siguiente mensaje |
| `claude.py` | 63 | Solo texto; si modelo solo usa tools | `"(Sin respuesta de texto.)"` |
| `deterministic_handlers.py` | 855-862 | No comprueba `created: false` del API | Siempre «Tarea creada:» |
| `deterministic_handlers.py` | 872-878 | Igual para hábitos | Siempre «Hábito creado:» |
| `deterministic_handlers.py` | 756-758 | Payload no parseable | JSON truncado crudo (~450 chars) |
| `deterministic_handlers.py` | 205 | `_format_amount` → EUR | Importes en EUR en Suiza (CHF) |
| `atlas_client.py` | 111 | `GET .../health/emotional/latest/` | 404 si se usara `context.py` |
| `atlas_client.py` | 117 | `GET .../exercise/recent/` | 404 |
| `atlas_client.py` | 428 | `POST .../health/exercise/log/` | 404 (correcto: `exercise/log/`) |
| `atlas_client.py` | 439 | `POST .../finance/patrimony/snapshot/create/` | 404 (correcto: `finance/patrimony/`) |
| `conversation_flow.py` | 150-151 | Memoria: `except Exception` → log | Usuario no sabe que falló el resumen |
| `daily_automation.py` | 243-244 | Fallo envío Telegram | Solo log |

**Sospecha que requiere runtime:** Claude dice «listo» sin tool MCP ejecutada, o ignora error en `mcp_tool_result`. El código solo extrae bloques `text` (`claude.py:53-58`) y no valida éxito de tools.

### Confirmaciones post-escritura

| Ruta | Calidad |
|------|---------|
| Determinista gasto/tarea/hábito | Buena: título, fecha, importe, excerpt API |
| Determinista duplicados | Mala: dice «creada» aunque API devuelva duplicado |
| Agente Claude | Variable: prompts piden confirmar **antes**, no después con dato concreto |
| MCP `create_task` duplicado | Devuelve `created: false` (`mcp_app/server.py:412-414`) — el usuario solo lo ve si Claude lo cita |

### Desincronización URLs (bot REST vs Atlas Vital)

| Bot (`atlas_client.py`) | Atlas Vital (`life/urls.py`) | Estado |
|-------------------------|------------------------------|--------|
| `/api/assistant/health/emotional/latest/` | No existe | **404** |
| `/api/assistant/exercise/recent/` | No existe | **404** |
| `/api/assistant/health/exercise/log/` | `/api/assistant/exercise/log/` | **404** |
| `/api/assistant/finance/patrimony/snapshot/create/` | `/api/assistant/finance/patrimony/` | **404** |
| Resto de paths en `atlas_client.py` | Coinciden con `life/urls.py:298-413` | OK |

**Endpoints Atlas sin equivalente en bot/MCP:** `trash`, `body/*`, `projects/*`, `time-logs/*`, `work/*`, `reviews/yearly/*`, `profile`, `habits/today` (separado).

### ¿Alucina acciones?

| Mecanismo | Verificado en código |
|-----------|----------------------|
| Ruta determinista | No: solo HTTP tras parseo |
| Ruta agente | Tool use estructurado vía MCP; texto final no ligado al resultado |
| Parseo de texto libre para acciones | No existe |
| Prompts «confirma antes de crear» | Fricción extra; puede preguntar sin haber llamado tool aún |

### Manejo de errores

- `atlas_client._get/_post`: `raise_for_status()` — excepción sube.
- Determinista: capturada en `conversation_flow.py:176-181` → mensaje genérico.
- Agente Anthropic: `conversation_flow.py:210-215` → mensaje genérico.
- **Sin reintentos** en HTTP ni Claude.
- Logs `logger.exception` existen pero no hay alertas ni dashboard.

---

## Fase 4 — Fricción: web vs bot

| Acción | Pasos en web (Atlas Vital) | Turnos en bot | Veredicto |
|--------|---------------------------|---------------|-----------|
| Marcar hábito | 1 clic en timeline (`execution.py:2748`) | 1 si sintaxis exacta; 2-3 si lenguaje natural | **Web gana** |
| Registrar gasto | Formulario con categoría | 1 turno sin categoría; varios si natural | **Web gana** |
| Ver agenda hoy | Panel visible al abrir | 1 turno `qué tengo hoy` | **Empate** |
| Completar tarea | 1 clic | 1-2 con `completa tarea X` | **Web gana** |
| Crear tarea con hora | Modal visual | Sintaxis o error de parseo | **Web gana** |

**Información que el bot podría inferir pero pide o no usa:**

- Fecha de hoy (a veces exige fecha explícita para tareas sin hora, `deterministic_handlers.py:627-628`)
- ID de hábito cuando hay match parcial múltiple
- `category_id` en gastos (MCP lo soporta; determinista no)

**Web puede, bot no (fuerza abrir web):**

- Plantillas de día
- Time logs, proyectos, basura/restaurar
- Medidas corporales (`body/create`)
- Review anual interactiva
- Toggle visual sin recordar nombres exactos

**Ceremonia innecesaria:**

- Mensaje «Conectando con tu Coach…» en cada cambio de agente
- Menú `ayuda` de 20+ líneas
- Confirmación previa en prompts para acciones que la web hace en 1 clic
- Respuestas largas por diseño (3-4 párrafos en prompts)

---

## Top 5 arreglos priorizados (impacto / esfuerzo)

### 1. Cablear o eliminar contexto dinámico — Esfuerzo: **M**

**Qué cambiar:** Llamar `fetch_context_for_agent` en `conversation_flow.py` antes de `generate_with_tools`, o quitar las promesas de contexto de los prompts. Corregir las 4 URLs rotas en `atlas_client.py`.

**Archivos:** `conversation_flow.py`, `atlas_client.py`, `prompts/agents/*.md`

### 2. Confirmaciones con evidencia del API — Esfuerzo: **M**

**Qué cambiar:** Tras cada escritura (determinista y post-MCP), mostrar eco del JSON Atlas (`id`, título, `created: true/false`). En ruta agente, no aceptar respuestas de escritura sin tool call verificable.

**Archivos:** `deterministic_handlers.py`, `claude.py`

### 3. Expandir ruta determinista al lenguaje cotidiano — Esfuerzo: **M**

**Qué cambiar:** Reconocer «ya hice X», «he gastado X en Y», «tarea hecha X» sin prefijos rígidos. Inferir fecha de hoy y único hábito coincidente.

**Archivos:** `deterministic_handlers.py`

### 4. Reducir fricción de agentes — Esfuerzo: **S**

**Qué cambiar:** Router menos agresivo (no cambiar agente por una keyword suelta); eliminar mensaje de transición; quitar «confirma antes» en prompts para acciones de 1 clic (marcar hábito, registrar gasto).

**Archivos:** `router.py`, `conversation_flow.py`, `prompts/agents/*.md`

### 5. Router barato (Haiku) + Sonnet solo si hace falta — Esfuerzo: **L**

**Qué cambiar:** Clasificar intent con modelo barato; enrutar CRUD simple a REST determinista ampliado; reservar Sonnet + MCP para coaching/reflexión.

**Archivos:** `conversation_flow.py`, nuevo módulo de routing LLM

---

## Cómo probar lo no verificable sin runtime

1. Enviar «ya hice meditación» y comprobar en Atlas Vital si `HabitLog` se creó + si la respuesta citó tool MCP (logs `stop_reason` en `claude.py:46-50`).
2. `curl` a las 4 URLs rotas de `atlas_client.py` contra Atlas Vital → confirmar 404.
3. Crear tarea duplicada vía «crea tarea X hoy» dos veces → ver si el bot dice «creada» ambas veces.
4. Medir latencia p50/p95: mensaje determinista vs. mensaje libre (timestamps en logs).

---

## Glosario de rutas de integración

| Carril | Auth | Tools / REST |
|--------|------|--------------|
| REST (`atlas_client.py`) | Header `X-Assistant-Key` | Usado por determinista y automatizaciones |
| MCP (`claude.py` → `/mcp`) | Bearer token (`authorization_token`) | Usado por ruta agente Claude |
| Django MCP server | `atlas/asgi.py:41` monta `/mcp` | `mcp_app/server.py` — ORM directo, sin HTTP interno |

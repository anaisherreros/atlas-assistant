# Plan de evolución: Atlas → segundo cerebro ("Jarvis")

**Fecha:** 2026-07-11
**Alcance:** `atlas-assistant` (bot Telegram) + `atlas-vital` (Django). Este documento vive aquí pero cubre ambos repos.
**Criterio rector:** la visión conceptual de Anaïs (2026-07-11). Toda función propuesta ha pasado el filtro de las 7 preguntas (¿reduce carga mental?, ¿reduce decisiones?, ¿ayuda a concentrarse?, ¿integra o crea silo?, ¿protege capacidad?, ¿devuelve info cuando es útil?, ¿quién gestiona a quién?).
**Complementa:** `AUDITORIA_ESTADO.md` (2026-06-12), que sigue vigente como diagnóstico técnico.

---

## Resumen ejecutivo

El sistema actual es un CRUD conversacional competente: Telegram → Claude + MCP → Django. Para convertirlo en segundo cerebro faltan **cinco sustantivos en el modelo de datos** (Idea, Decisión, Contexto Vital, Aparcado, Nivel de capacidad) y **tres comportamientos en el bot** (nunca mentir sobre escrituras, una sola identidad, capturar sin comprometer).

El plan tiene 6 fases ordenadas por dependencia y por impacto en carga mental. Cada fase es utilizable por sí sola; ninguna requiere la siguiente.

| Fase | Nombre | Repos | Esfuerzo | Migraciones DB |
|------|--------|-------|----------|----------------|
| 1 | Confianza + un solo Atlas | assistant | S–M | No |
| 2 | Captura universal (Inbox) | ambos | M | Sí |
| 3 | Contexto Vital | ambos | M | Sí |
| 4 | Memoria de decisiones + búsqueda | ambos | L | Sí |
| 5 | Capacidad y retorno | ambos | M | Sí |
| 6 | Patrones e insights | assistant | S | No |

---

## Arquitectura objetivo

```
  Telegram (voz/texto)          Web Atlas Vital (HTMX)
        │  interfaz rápida            │  gestión profunda
        ▼                             ▼
  ┌─────────────────────────────────────────────┐
  │                 ATLAS (una identidad)       │
  │  atlas-assistant:                           │
  │   · tiering Haiku (CRUD) / Sonnet (reflex.) │
  │   · Contexto Vital inyectado en cada turno  │
  │   · captura ≠ compromiso                    │
  └──────────────────────┬──────────────────────┘
                         │ MCP (único carril)
                         ▼
  ┌─────────────────────────────────────────────┐
  │  atlas-vital (Django) — fuente de verdad    │
  │  Tareas · Hábitos · Deseos · Finanzas ·     │
  │  Journal · Revisiones · INBOX · DECISIONES ·│
  │  CONTEXTO VITAL · índice semántico          │
  └─────────────────────────────────────────────┘
```

Principio: **un solo cerebro, dos interfaces, una fuente de verdad.** El carril REST (`atlas_client.py` + `X-Assistant-Key`) se mantiene solo para la ruta determinista y las automatizaciones; todo lo nuevo entra por MCP.

---

## FASE 1 — Confianza + un solo Atlas

**Objetivo:** que puedas soltar carga mental porque lo que Atlas dice que hizo, lo hizo. Y que hables con Atlas, no con una centralita.
**Solo `atlas-assistant`. Sin migraciones. Es la base de todo lo demás.**

### 1.1 Nunca decir "hecho" sin evidencia

- `deterministic_handlers.py`: leer `created`/`deleted` del JSON de respuesta. Si `created: false` → «Ya existía: *título* (#id)», nunca «Tarea creada». (Líneas ~855-878 según auditoría.)
- `claude.py`: capturar los bloques `mcp_tool_use`/`mcp_tool_result` de la respuesta, no solo el texto. Devolver junto al texto una lista `tools_used` con éxito/fallo.
- `conversation_flow.py`: si el mensaje del usuario pide una escritura (heurística barata: verbos crea/registra/apunta/marca/completa) y la respuesta no contiene ningún tool call de escritura, reintentar una vez con instrucción explícita; si vuelve a fallar, decir honestamente «no he podido registrarlo».
- Toda confirmación de escritura cita el dato real del API: `#id`, título, fecha, `created: true/false`.
- Un reintento con backoff en llamadas HTTP (`atlas_client.py`) y Anthropic.

### 1.2 Una sola identidad: Atlas

- Eliminar el cambio de agente por keywords en `router.py` y el mensaje «Conectando con tu Coach…» (`conversation_flow.py:168-170`).
- Fusionar `prompts/agents/{personal,coach,performance,financial}.md` en **un solo `prompts/atlas.md`** con secciones de tono: registro (seco, 1 línea), reflexión (estilo coach, confrontación amable), finanzas (sin inventar datos), rendimiento (nutrición antiinflamatoria, lesiones). El modelo modula el tono; la usuaria nunca "cambia de agente".
- Mantener y afinar `model_tiering.py` (ya existe): Haiku para CRUD/captura, Sonnet para reflexión/análisis. Esto implementa el principio de "profundidades distintas".
- La columna `active_agent` de la DB del bot queda obsoleta: fijar a `atlas` (sin migración destructiva; se ignora).
- Quitar de los prompts el «confirma antes de escribir» para acciones de 1 clic (marcar hábito, gasto simple). Confirmar **después** con evidencia sustituye a preguntar antes.
- Acortar el menú `ayuda` a ~6 líneas.

### 1.3 Limpieza de código muerto

- Borrar: `tools.py` (dispatch), `message_classification.py`, `agent_tool_policy.py`, `agents/nutritionist.py`, `agents/trainer.py`.
- `agents/context.py`: **no borrar** — se recicla en Fase 3 como base del inyector de Contexto Vital (sus 4 URLs rotas ya fueron corregidas).

### Criterios de hecho (Fase 1)

1. Crear la misma tarea dos veces → la segunda vez el bot dice «ya existía», no «creada».
2. Decir «gym» en mitad de una conversación → no hay cambio de agente ni mensaje de transición.
3. `grep` de los módulos muertos → cero referencias, archivos eliminados.
4. Pedir «registra X» y forzar un fallo del API → el bot dice que falló, no que está hecho.

---

## FASE 2 — Captura universal (Inbox)

**Objetivo:** «Dime lo que tienes en la cabeza. Yo me encargo.» Capturar cuesta cero y **nunca** convierte un pensamiento en obligación.
**Es el corazón de la visión (principios 2, 3 y 16).**

### 2.1 Modelo `InboxItem` (`life/models/inbox.py`)

```python
class InboxItem(models.Model):
    KIND_CHOICES = [
        ("idea", "Idea"),                    # p. ej. idea de contenido
        ("interes_futuro", "Interés futuro"),# "algún día podría…"
        ("nota", "Nota / referencia"),
        ("posible_tarea", "Posible tarea"),  # suena a tarea pero no confirmada
        ("sin_clasificar", "Sin clasificar"),
    ]
    STATUS_CHOICES = [
        ("nueva", "Nueva"),
        ("clasificada", "Clasificada"),
        ("aparcada", "Aparcada"),        # con review_at
        ("convertida", "Convertida"),    # ya es tarea/deseo/proyecto
        ("descartada", "Descartada"),
    ]
    user = FK(User)
    raw_text = TextField()               # literal, lo que dijo
    summary = CharField(200, blank=True) # título corto generado
    kind = CharField(choices=KIND_CHOICES, default="sin_clasificar")
    status = CharField(choices=STATUS_CHOICES, default="nueva")
    area = FK(Area, null=True, blank=True)
    project = FK(ProjectProfile, null=True, blank=True)
    review_at = DateField(null=True, blank=True)   # "recuérdamelo en septiembre"
    source = CharField(choices=[web|telegram|voz])
    converted_task = FK(Task, null=True); converted_desire = FK(Desire, null=True)
    created_at / updated_at
```

### 2.2 Tools MCP nuevas (`mcp_app/server.py`)

- `capture_inbox(text, kind_hint="", area="", review_at="")` — crear item. **Rápida y sin preguntas.**
- `list_inbox(status="nueva", limit=20)`
- `classify_inbox_item(item_id, kind, area="")`
- `park_inbox_item(item_id, review_at)` — aparcar con fecha
- `convert_inbox_item(item_id, to="task"|"desire", **campos)` — marca `convertida` y enlaza
- `discard_inbox_item(item_id)`

### 2.3 Comportamiento del bot (la parte crítica)

Regla de clasificación en el prompt + tiering:

- «Se me ha ocurrido…», «apunta esto», «idea:», «algún día…», «me interesa…», «tal vez debería…» → **`capture_inbox` con Haiku**, respuesta de 1 línea: *«Guardada como idea (#12, Proyectos). Sigue a lo tuyo.»* **Prohibido preguntar** fecha, prioridad o si lo convierte en tarea.
- «Tengo que…», «recuérdame…», «mañana a las 10…» → tarea normal (flujo actual).
- Duda genuina → `capture_inbox` como `sin_clasificar`. **En caso de duda, capturar sin comprometer** — nunca al revés.
- «Recuérdamelo en septiembre» → `park_inbox_item(review_at="2026-09-01")`.

### 2.4 Web + digest

- Vista Inbox en Atlas Vital (lista con acciones clasificar / convertir / aparcar / descartar, HTMX como el resto).
- Digest matutino: si hay items `nueva` de hace >48 h, incluir **máximo 3** con propuesta de clasificación. Items `aparcada` cuya `review_at` llega hoy → aparecen en el digest de ese día. Nada de pushes sueltos.

### Criterios de hecho (Fase 2)

1. «Se me ha ocurrido un vídeo sobre Saturno en casa 5» → 1 respuesta de 1 línea, cero preguntas, item visible en la web.
2. «Recuérdamelo en septiembre» → item aparcado que reaparece en el digest del 1 de septiembre.
3. Ninguna captura genera tarea, fecha ni prioridad automáticamente.

---

## FASE 3 — Contexto Vital

**Objetivo:** no tener que explicarle tu vida en cada conversación (principio 8).

### 3.1 Modelo `VitalContext` (singleton por usuario)

Secciones de texto libre editables (mejor que campos rígidos — la vida no cabe en un schema):

```python
class VitalContext(models.Model):
    user = OneToOneField(User)
    trabajo = TextField(blank=True)          # trabajo actual, %, horarios
    formacion = TextField(blank=True)
    prioridades = TextField(blank=True)      # máx 3 focos, en texto
    proyectos = TextField(blank=True)        # activos y pausados
    restricciones = TextField(blank=True)    # energía, salud, límites
    etapa_vital = TextField(blank=True)
    decisiones_vigentes = TextField(blank=True)  # resumen; detalle en Fase 4
    updated_at = DateTimeField(auto_now=True)
```

### 3.2 Integración

- Página en la web para editarlo (formulario simple, tema oscuro como el resto).
- Tools MCP: `get_vital_context()`, `update_vital_context(section, text)`.
- Bot: inyectar el Contexto Vital completo (~300-500 tokens) + agenda de hoy en el system prompt de **cada** turno. Reciclar la infraestructura de `agents/context.py` que hoy está desconectada.
- El bot **propone** actualizaciones cuando detecta cambios («has mencionado que reduces jornada — ¿actualizo tu Contexto Vital?») pero **nunca lo edita solo**. Es tu fotografía; tú la firmas.
- La memoria conversacional actual (resumen cada 20 mensajes) queda relegada a color conversacional; el Contexto Vital pasa a ser la fuente de "quién eres ahora".

### Criterios de hecho (Fase 3)

1. Preguntar «¿qué debería hacer mañana?» → la respuesta refleja prioridades del Contexto Vital sin que las menciones.
2. Cambiar una prioridad en la web → la siguiente conversación de Telegram ya la conoce.
3. El bot jamás modifica el Contexto Vital sin confirmación explícita.

---

## FASE 4 — Memoria de decisiones + búsqueda semántica

**Objetivo:** «¿por qué habíamos decidido esto?» tiene respuesta (principio 7), y todo lo escrito es recuperable (principio 18).

### 4.1 Modelo `Decision`

```python
class Decision(models.Model):
    user = FK(User)
    title = CharField(200)               # "Astrología como tema inicial del canal"
    decided_on = DateField()
    reasoning = TextField()              # por qué
    alternatives = TextField(blank=True) # qué se consideró y descartó
    revisit_when = TextField(blank=True) # qué información justificaría revisarla
    status = CharField(choices=[vigente|revisada|revocada], default="vigente")
    superseded_by = FK("self", null=True)
    area = FK(Area, null=True); project = FK(ProjectProfile, null=True)
```

Tools: `log_decision(...)`, `list_decisions(status="vigente")`, `get_decision(id)`, `revise_decision(id, new_reasoning)`.

Bot: al detectar «he decidido…», «al final voy a…» propone registrarla (una sola pregunta: «¿La guardo como decisión con este porqué: …?»). Al detectar duda sobre algo ya decidido («no sé si dejar la astrología…») → recupera la decisión vigente, presenta el porqué original y pregunta «¿ha aparecido información nueva desde entonces?». **Contexto, no psicoanálisis.**

### 4.2 Búsqueda semántica (`search_my_life`)

- **Qué se indexa:** `JournalEntry`, `Decision`, revisiones (diaria/semanal/mensual/anual), `InboxItem`, notas de proyecto (`ProjectUpdate`).
- **Cómo:** tabla `ContentEmbedding(content_type, object_id, chunk_text, vector JSONField, updated_at)`. Embeddings vía API (OpenAI `text-embedding-3-small` — la key ya existe para Whisper; alternativa Voyage). Búsqueda por coseno **en Python, fuerza bruta**: a escala personal (miles de filas) es <100 ms y funciona idéntico en SQLite (dev) y Postgres (Railway) sin extensiones. Si algún día duele, migrar a pgvector es un cambio local.
- Indexación: señal `post_save` o comando de management nocturno.
- Tool MCP: `search_my_life(query, kinds=[], date_from="", date_to="")` → devuelve pasajes con fecha y origen.
- Solo se activa con Sonnet (mensajes reflexivos); las capturas y CRUD no la tocan.

### Criterios de hecho (Fase 4)

1. «¿Por qué decidimos empezar por astrología?» → respuesta con la decisión registrada, fecha y porqué.
2. «¿Qué escribí sobre el gimnasio en marzo?» → pasajes reales del journal/revisiones, citados con fecha.
3. «Analiza los últimos 6 meses y dime por qué no avanzo» → usa datos reales recuperados, no generalidades.

---

## FASE 5 — Capacidad y retorno

**Objetivo:** Atlas ayuda a hacer *menos* (principios 5, 10, 17).

### 5.1 Capacidad finita

- `VitalContext` gana `max_focos = 3` (configurable).
- Tool `check_capacity()` → focos activos vs. límite.
- Prompt: ante «quiero empezar X», Atlas consulta capacidad y responde con las opciones reales: *añadir / sustituir / pausar / aparcar como interés futuro*. Nunca añade en silencio.

### 5.2 Aparcar con confianza

- `Desire` y `ProjectProfile` ganan estado `pausado` + `review_at` opcional (el Inbox ya lo tiene de Fase 2).
- «Pausa el proyecto X» por Telegram → pausado también en la web (misma DB, ya garantizado por MCP).
- Lo aparcado **desaparece** de las vistas activas y del contexto inyectado — proteger atención (principio 4) — y reaparece solo cuando llega su `review_at`.

### 5.3 Niveles mínimo / normal / expansivo

- `Habit` gana `nivel_minimo`, `nivel_normal`, `nivel_expansivo` (CharField descriptivos, ej. «1 sesión ligera» / «2 sesiones» / «extra si hay energía»).
- `HabitLog` gana `nivel` (default `normal`). Marcar el mínimo cuenta como continuidad, sin asterisco de culpa.

### 5.4 Sin rachas, con retorno

- Auditar UI y digests: eliminar todo lenguaje de racha/cadena/fracaso si existe.
- Si un hábito frecuente lleva ≥7 días sin logs → **una** línea en el digest matutino: «¿versión mínima de X para volver hoy?». Sin repetición diaria machacona; si se ignora 2 veces, se calla una semana.

### Criterios de hecho (Fase 5)

1. «Quiero empezar a aprender medicina china» con 3 focos activos → Atlas ofrece aparcar o sustituir, no añade.
2. Un hábito marcado en mínimo 5 días seguidos se muestra como continuidad, no como fallo.
3. Nada aparcado aparece en «¿qué tengo hoy?».

---

## FASE 6 — Patrones e insights

**Objetivo:** Atlas aprende cómo funcionas (principio 11), sin inventar psicología.

- Job semanal (domingo, antes de la revisión semanal) con Sonnet sobre datos agregados de 4-8 semanas: hábitos por día/contexto, energía vs. turnos de trabajo, tareas pospuestas ≥3 veces, ideas recurrentes en el Inbox.
- Formato de salida **obligatorio y fijo**:
  - `HECHO:` «En 8 de las últimas 10 semanas, el segundo entrenamiento no se realizó tras 4 turnos seguidos.»
  - `POSIBLE LECTURA (pregunta):` «¿Ese hueco es realista o lo movemos?»
- Máximo 3 hechos + 2 preguntas. Se entrega **dentro** del flujo de revisión semanal existente, jamás como notificación aparte.

---

## Qué NO se construye (vetado por el filtro de la visión)

| Tentación | Por qué no |
|-----------|------------|
| Más agentes visibles / personalidades | Principio 13: una identidad. Ya se eliminan en Fase 1. |
| Dashboards nuevos de estadísticas | Acumula sin devolver; la web ya muestra lo necesario. |
| Notificaciones push fuera de los 2 digests | Rompe protección de atención (principio 4). |
| Rachas, puntos, gamificación | Principio 17: diseño para volver, no para no fallar. |
| Categorización obligatoria al capturar | Mata el bucle central de captura (principio 2). |
| Auto-conversión idea→tarea "inteligente" | Principio 3: la creatividad no se convierte sola en obligación. |

---

## Decisiones abiertas (para decidir sobre la marcha, no bloquean la Fase 1)

1. **Embeddings:** OpenAI (key ya existente) vs. Voyage. Propuesta: OpenAI por simplicidad.
2. **Ubicación del Inbox en la web:** sección propia vs. dentro de Ejecución. Propuesta: sección propia en la navegación principal (es un concepto central, no un apéndice).
3. **Idioma de los campos nuevos:** el código actual mezcla inglés (modelos) y español (UI). Propuesta: seguir la convención existente (modelos en inglés, choices/UI en español).
4. **La ruta determinista** (frases exactas): mantenerla como atajo de latencia cero, pero revisarla en Fase 2 para que también capture («apunta X» sin LLM).

## Riesgos

- **Fase 1 toca el flujo principal del bot** → hacerla en rama, probar los 4 criterios de hecho antes de desplegar.
- **Migraciones en producción (Railway/Postgres)** desde Fase 2 → backup antes de cada `migrate`; los modelos nuevos son aditivos (riesgo bajo).
- **Coste API:** la búsqueda semántica y los insights usan Sonnet; el tiering existente ya limita el gasto. Los embeddings son centavos a escala personal.
- **Alcance:** cada fase se cierra y se usa antes de empezar la siguiente. Si una fase lleva usándose 2 semanas y algo sobra, se recorta antes de seguir — el propio plan pasa por el filtro de las 7 preguntas.

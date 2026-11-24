# Inferencia, API y monitorización

> Secciones a cargo de Derlis Daniel. Borrador para integrar en el `README.md`.
> Las cifras de probabilidad corresponden al modelo `regresion_logistica`
> versión 1; si cambia el modelo final, se regeneran con los mismos comandos.

Todas las vías de predicción **reutilizan el mismo pipeline guardado**
(`artifacts/model.joblib`, preprocesamiento + modelo). Ninguna reentrena. Si el
artefacto no existe, se genera con:

```bash
uv run python -m mlops_course.train
```

## Contrato de entrada

13 variables con los nombres del dataset. `target` **nunca** es una entrada.

| Tipo | Variable | Valores admitidos |
|---|---|---|
| Numérica | `age` | 1 – 120 |
| Numérica | `trestbps` | 50 – 250 |
| Numérica | `chol` | 50 – 700 |
| Numérica | `thalach` | 50 – 250 |
| Numérica | `oldpeak` | 0 – 10 |
| Categórica | `sex` | 0, 1 |
| Categórica | `cp` | 1, 2, 3, 4 |
| Categórica | `fbs` | 0, 1 |
| Categórica | `restecg` | 0, 1, 2 |
| Categórica | `exang` | 0, 1 |
| Categórica | `slope` | 1, 2, 3 |
| Categórica (opcional) | `ca` | 0, 1, 2, 3 o vacío |
| Categórica (opcional) | `thal` | 3, 6, 7 o vacío |

- Los rangos numéricos son límites de **plausibilidad** para detectar errores de
  captura o de unidades, no criterios clínicos. Se definen una sola vez en
  `schemas.py` y los usan la API y el batch.
- `ca` y `thal` pueden faltar, como en el dataset original: el pipeline los
  imputa con lo aprendido en entrenamiento.
- **Por qué se validan los valores:** el pipeline usa
  `OneHotEncoder(handle_unknown="ignore")`, así que un código desconocido
  (p. ej. `thal = 5`) no daría error, sino una predicción silenciosamente
  incorrecta. La validación lo impide.

## Inferencia batch

```bash
uv run python -m mlops_course.predict                                   # ejemplo válido
uv run python -m mlops_course.predict --input data/ejemplos/pacientes_invalidos.csv \
    --output reports/predicciones_invalidos.csv                         # ejemplo inválido
uv run python -m mlops_course.predict --input data/ejemplos/pacientes_sin_columna.csv
```

- Entrada por defecto: `data/ejemplos/pacientes_validos.csv`.
- Salida por defecto: `reports/predicciones.csv`.
- La salida conserva **todas las filas, en el mismo orden**, y añade
  `probabilidad` (de la clase 1), `prediccion` (0 sin enfermedad / 1 con
  enfermedad), `error`, `umbral` y `model_version`.

Los ejemplos se generan de forma reproducible con
`uv run python scripts/crear_ejemplos.py`. Las filas proceden del **conjunto de
test** (misma partición 80/20, semilla 42), no de entrenamiento.

**Dos niveles de error:**

| Tipo | Ejemplo | Comportamiento |
|---|---|---|
| Estructura | falta o sobra una columna, viene `target`, el archivo no existe | Se detiene con un mensaje claro (código de salida 1) |
| Fila | `thal = 5`, edad negativa, texto en un número, obligatorio vacío | Solo esa fila queda sin predicción, con el motivo en `error`; el resto se predice |

Resultado del ejemplo inválido:

```text
Modelo versión 1 | 5 filas leídas, 1 predichas, 4 con error
   probabilidad  prediccion                                          error
0           NaN        <NA>  thal no admitido (valores válidos: [3, 6, 7])
1           NaN        <NA>                    age fuera de rango [1, 120]
2           NaN        <NA>                           chol no es un número
3           NaN        <NA>                sex es obligatorio y está vacío
4        0.3132           0
```

Resultado del CSV sin columna:

```text
ERROR: Columnas incorrectas: faltan ['thal']. Se esperan exactamente estas 13: [...]
```

## API HTTP

```bash
uv run uvicorn mlops_course.api:app --host 127.0.0.1 --port 8000
```

Documentación interactiva: <http://127.0.0.1:8000/docs>. El modelo se carga
**una sola vez al arrancar**, no en cada petición.

| Endpoint | Respuesta |
|---|---|
| `GET /health` | `200 {"status": "ok", "model_version": "1"}` · `503` si no hay modelo |
| `POST /predict` | `200` con `prediccion`, `probabilidad`, `umbral`, `model_version` · `422` entrada inválida · `503` sin modelo |

### Ejemplo válido

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @data/ejemplos/paciente_valido.json
```

```json
{"prediccion": 0, "probabilidad": 0.267, "umbral": 0.5, "model_version": "1"}
```

Es el mismo paciente que la fila 0 de `pacientes_validos.csv`: el batch da
**la misma probabilidad (0.267)**, lo que confirma que ambas vías usan el mismo
pipeline.

### Ejemplo inválido

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @data/ejemplos/paciente_invalido.json
```

```json
{
  "detail": "La entrada no cumple el contrato de la API",
  "errores": [
    {"campo": "age",    "valor_recibido": -3,   "problema": "fuera de rango: debe ser >= 1"},
    {"campo": "sex",    "valor_recibido": null, "problema": "campo obligatorio ausente"},
    {"campo": "thal",   "valor_recibido": 5,    "problema": "valor no admitido; valores válidos: 3, 6, 7"},
    {"campo": "target", "valor_recibido": 1,    "problema": "'target' no puede enviarse: es lo que el modelo predice"}
  ]
}
```

El 422 por defecto de FastAPI es técnico y en inglés. Se sustituye por un
formato que indica **qué campo falla y por qué**, y se documenta en `/docs`.

Si falta el artefacto, la API arranca igualmente y responde `503` con el
comando para generarlo, en lugar de caerse sin explicación.

## Registro de inferencias

Cada ejecución del batch y cada petición a `/predict` añade una fila a
`logs/inferencias.csv`:

```text
fecha,origen,model_version,n_registros,n_predichos,n_errores,tipo_error,campos_con_error
2026-11-24T16:32:19+00:00,batch,1,5,5,0,,
2026-11-24T16:32:23+00:00,batch,1,5,1,4,validacion_fila,age;chol;sex;thal
2026-11-24T16:32:26+00:00,batch,,0,0,1,estructura,
2026-11-24T16:33:05+00:00,api,1,1,0,1,validacion,age;sex;target;thal
```

- **No se guarda ningún valor clínico**: solo *qué campos* fallaron, nunca su
  valor.
- Fecha en UTC, para no depender de la zona horaria del servidor.
- Se registran también los fallos (validación, estructura, `sin_modelo`): un
  registro solo de éxitos no sirve para monitorizar.
- Si no se puede escribir el registro, se avisa, pero **la predicción no
  falla**.

## Monitorización y drift

```bash
uv run python -m mlops_course.monitoring simular --escenario estable   # control
uv run python -m mlops_course.monitoring informe
uv run python -m mlops_course.monitoring simular                       # cambio controlado
uv run python -m mlops_course.monitoring informe
```

Salidas: `reports/drift_report.html` (informe interactivo de Evidently) y
`reports/drift_resumen.md` (tabla por variable).

### Diseño

- **Referencia:** las 193 filas de **entrenamiento** (lo que el modelo conoce).
- **Lote actual:** una **copia** de las 61 filas de **test** (datos
  reservados), **sin `target`**, como llegarían los pacientes en producción.
  El CSV original y el test de evaluación no se modifican.
- **Cambio controlado** (escenario `cambio`): **edad +10 años** y **colesterol
  ×1,15**, que simula una población de más edad y con colesterol más alto (por
  ejemplo, otro centro). Los valores se mantienen dentro del contrato de
  entrada.
- **Escenario de control** (`estable`): la misma copia **sin cambios**. Sirve
  para distinguir las alertas causadas por el cambio de las que aparecen sin
  él.
- No se remuestrea el lote para inflar su tamaño: repetir 61 filas no aporta
  información y haría que los tests parecieran más concluyentes de lo que son.
- Se vigilan las 13 variables **y la probabilidad predicha**.

### Método y criterio

Evidently `DataDriftPreset`, con sus valores por defecto para muestras pequeñas:

- Numéricas: test de **Kolmogórov-Smirnov**. Drift si **p-valor < 0,05**.
- Categóricas: **chi-cuadrado** (Z-test si son binarias). Drift si
  **p-valor < 0,05**.
- Conjunto: drift si al menos el **50 %** de las columnas lo presentan.

### Resultados

| Columna | Control (sin cambio) | Con cambio | Medias ref. → actual (cambio) |
|---|---|---|---|
| `age` | no (p = 0,73) | **sí** (p < 0,001) | 54,5 → 64,0 |
| `chol` | no (p = 0,054) | **sí** (p = 0,025) | 248,2 → 269,3 |
| `restecg` | **sí** (p < 0,001) | **sí** (p < 0,001) | — |
| `ca` | **sí** (p = 0,022) | **sí** (p = 0,022) | — |
| `probabilidad` | no | no (p = 0,49) | 0,461 → 0,529 |
| Resto (9 columnas) | no | no | — |
| **Columnas con drift** | 2 de 14 | 4 de 14 (28,6 %) | drift del conjunto: **no** |

### Interpretación

- **El cambio se detecta.** Las únicas alertas **nuevas** respecto al control
  son `age` y `chol`, justo las variables modificadas.
- **`restecg` y `ca` son falsas alarmas**, porque aparecen también sin ningún
  cambio:
  - `restecg = 1` aparece **1 vez** en entrenamiento y **3 veces** en el lote.
    Con categorías tan poco frecuentes, el chi-cuadrado no es fiable.
  - `ca` difiere por variación natural entre dos muestras pequeñas (por
    ejemplo, 3 vacíos en el lote frente a 1 en entrenamiento).
  - Además, con 14 tests a un nivel de 0,05 es esperable alguna alerta por
    azar.
  - Sin el escenario de control, estas alertas se habrían atribuido
    erróneamente al cambio introducido.
- **`chol` se detecta por muy poco** (p = 0,025; en el control ya estaba en
  0,054). Con 61 filas, los tests tienen poca potencia: un cambio de un 15 %
  apenas se distingue de la variación natural.
- **La salida del modelo se desplaza** (probabilidad media de 0,46 a 0,53),
  aunque sin superar el umbral de drift: el modelo tiende a asignar más riesgo
  a esta población.
- **Drift no es pérdida de rendimiento.** El lote no tiene etiquetas, así que
  **no se puede medir** si el modelo acierta más o menos. Un cambio en las
  entradas puede no afectar al rendimiento, o afectarlo mucho; sin etiquetas
  no se sabe.

### Actuación propuesta

1. **No reentrenar automáticamente.** Una alerta de drift no lo justifica: sin
   etiquetas no hay evidencia de que el modelo haya empeorado.
2. **Investigar el origen:** ¿hay una población nueva (otro centro, otro
   programa de cribado) o un error en los datos (unidades, captura)? Los
   valores siguen dentro del contrato, así que no parece un error de formato.
3. **Conseguir etiquetas** de una muestra del lote para **medir el rendimiento
   real** (recall, precisión) en la nueva población.
4. **Reentrenar solo si esa evaluación muestra degradación**, incluyendo datos
   de la nueva población y comparando el candidato con el modelo actual sobre
   los mismos datos.
5. **Mejorar el método:** agrupar categorías raras (como `restecg = 1`) o
   exigir un tamaño mínimo por categoría antes del chi-cuadrado, y acumular
   lotes más grandes antes de concluir.

### Limitaciones

- Muestras muy pequeñas (193 frente a 61): poca potencia y falsas alarmas en
  categorías poco frecuentes.
- Muchos tests a la vez sin corrección por comparaciones múltiples.
- El cambio es simulado; en un caso real la causa sería desconocida.

## Comprobaciones automáticas

```bash
uv run pytest tests/test_api.py -v
```

18 tests, **independientes de haber entrenado antes**: un *fixture* entrena un
pipeline con el mismo `build_pipeline` del proyecto, lo guarda con joblib en una
carpeta temporal y apunta la configuración allí. Así se prueba la **carga real
desde disco** sin tocar `artifacts/` ni `logs/`, y funcionan desde una copia
limpia del repositorio.

| Grupo | Qué comprueba |
|---|---|
| Carga del pipeline | Se carga desde disco y predice; si falta, el error explica cómo generarlo |
| API válida | `/health` 200 con versión; `/predict` 200 con clase, probabilidad y versión; acepta `ca`/`thal` vacíos |
| API inválida | 422 que señala el campo correcto: código no admitido, fuera de rango, texto, `target`, campo desconocido y obligatorio ausente |
| Sin modelo | `/health` y `/predict` responden 503 |
| Batch | Conserva filas y orden; marca filas inválidas sin detener el resto; rechaza columnas incorrectas |
| Coherencia | Batch y API dan la misma probabilidad para el mismo paciente |
| Registro | Columnas previstas y ninguna columna clínica |
| Contrato | `schemas.py` y `data.py` admiten los mismos códigos |

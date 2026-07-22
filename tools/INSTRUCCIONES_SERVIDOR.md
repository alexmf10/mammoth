# Gate de métricas y campaña Pareto en el servidor

Estas instrucciones están escritas para ejecutarse desde **Terminal de
JupyterLab** (menú `File` → `New` → `Terminal`). No uses una celda con
`!comando` para la campaña larga. `nohup` en la terminal permite cerrar el
navegador y apagar el ordenador local sin terminar el proceso. Un reinicio o
apagado administrativo del servidor sí puede detenerlo; en ese caso se usa el
procedimiento de reanudación del paso 5.

No ejecutes `pip install -e .`. El repositorio fijo es
`/home/amf380/mammoth`, el Python fijo es `/opt/environment/bin/python` y el
overlay de paquetes es `$HOME/.local/mammoth-pydeps`.

## 0. Verificar una sola vez el intérprete

**Qué hace:** comprueba que el kernel de Jupyter y los comandos de estas
instrucciones usan exactamente el mismo Python.

En una celda Python de Jupyter copia y ejecuta:

```python
import sys
print(sys.executable)
```

**Resultado esperado:**

```text
/opt/environment/bin/python
```

Después, en la Terminal de JupyterLab, copia este bloque:

```bash
cd /home/amf380/mammoth
export PYTHON_BIN="/opt/environment/bin/python"
export PYTHONPATH="$HOME/.local/mammoth-pydeps${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -c 'import sys; print(sys.executable)'
```

Debe volver a imprimir `/opt/environment/bin/python`. Si cualquiera de las
dos salidas es distinta, **detente** y devuelve ambas rutas; no sustituyas el
intérprete ni instales paquetes todavía.

> Cada terminal nueva pierde los `export`. Por eso los bloques importantes de
> abajo vuelven a incluirlos y se pueden copiar completos.

## 1. Actualizar `master`

**Qué hace:** descarga únicamente el trabajo versionado y comprueba qué commit
ha quedado activo.

```bash
cd /home/amf380/mammoth
export PYTHON_BIN="/opt/environment/bin/python"
export PYTHONPATH="$HOME/.local/mammoth-pydeps${PYTHONPATH:+:$PYTHONPATH}"
git status --short --branch
git pull --ff-only origin master
git rev-parse HEAD
```

**Cómo saber que fue bien:** `git pull` termina sin error y `git rev-parse`
imprime un SHA de 40 caracteres. La salida de `git status` no debe mostrar
cambios locales inesperados.

**Si falla:** no uses `reset`, `checkout` ni borres ficheros. Copia las tres
salidas completas y devuélvelas.

## 2. Protocolo de aceptación completo

No lances los 15 runs hasta completar todos estos subpasos. El smoke ejecuta
solo L2P, seed 0, una tarea y unas pocas iteraciones, pero pasa por el mismo
orquestador, wrapper CUDA, logging y validación que la campaña.

### 2.1. Guardar el snapshot del entorno

**Qué hace:** registra commit, GPU, driver, CUDA, rutas y versiones realmente
importadas de torch/torchvision/timm/kornia, precisión numérica, espacio y
cuota. No instala ni modifica nada.

```bash
cd /home/amf380/mammoth
export PYTHON_BIN="/opt/environment/bin/python"
export PYTHONPATH="$HOME/.local/mammoth-pydeps${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p results/campaign-acceptance
bash tools/env_snapshot.sh > results/campaign-acceptance/env_snapshot.txt 2>&1
SNAPSHOT_RC=$?
echo "env_snapshot_exit_code=$SNAPSHOT_RC"
tail -n 60 results/campaign-acceptance/env_snapshot.txt
```

**Cómo saber que fue bien:** aparece `env_snapshot_exit_code=0`; los imports
dicen `torch_import=OK`, `torchvision_import=OK`, `timm_import=OK` y
`kornia_import=OK`; las versiones esperadas son `torch==2.2.1`,
`torchvision==0.17.1`, `timm==0.9.8` y `kornia==0.7.1`. Comprueba tanto los
campos `*_distribution_version` como `*_module_version`, para detectar que no
se haya resuelto accidentalmente otra instalación. `sys.executable` debe ser
`/opt/environment/bin/python`, aunque `timm` y `kornia` deben resolverse desde
el overlay. La campaña no activa AMP: usa tensores FP32 con
`code_optimization=0`; los campos `*_allow_tf32` dejan registrado si algún
backend puede usar TF32 internamente.

**Si falla:** devuelve el fichero completo con:

```bash
cat /home/amf380/mammoth/results/campaign-acceptance/env_snapshot.txt
```

### 2.2. Arrancar el único sampler global de GPU

**Qué hace:** inicia un muestreo cada segundo de **todos** los procesos compute
que vea `nvidia-smi`, incluidos los de otros usuarios. Se mantiene durante el
smoke y toda la campaña. El CSV se añade, no se sobreescribe.

```bash
cd /home/amf380/mammoth
export PYTHON_BIN="/opt/environment/bin/python"
export PYTHONPATH="$HOME/.local/mammoth-pydeps${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p results/campaign
if [ -f results/campaign/gpu_sampler.pid ] && kill -0 "$(cat results/campaign/gpu_sampler.pid)" 2>/dev/null; then
  echo "El sampler ya estaba activo con PID $(cat results/campaign/gpu_sampler.pid)"
else
  nohup "$PYTHON_BIN" -u tools/gpu_sampler.py \
    --output results/campaign/gpu_samples.csv --interval 1 \
    > results/campaign/gpu_sampler.log 2>&1 < /dev/null &
  SAMPLER_PID=$!
  echo "$SAMPLER_PID" > results/campaign/gpu_sampler.pid
  echo "Sampler iniciado con PID $SAMPLER_PID"
fi
sleep 3
SAMPLER_PID="$(cat results/campaign/gpu_sampler.pid)"
ps -fp "$SAMPLER_PID"
head -n 5 results/campaign/gpu_samples.csv
```

**Cómo saber que fue bien:** `ps` muestra
`tools/gpu_sampler.py`; el CSV tiene la cabecera
`timestamp_utc,timestamp_epoch,pid,username,process_name,used_gpu_memory_mib,sample_status`
y al
menos una muestra. `no_compute_processes` es normal si la GPU estaba ociosa.

**Si falla:** devuelve:

```bash
cat results/campaign/gpu_sampler.log
tail -n 20 results/campaign/gpu_samples.csv 2>/dev/null
nvidia-smi
```

No inicies un segundo sampler si el primero sigue vivo.

### 2.3. Ejecutar el smoke de L2P mediante el orquestador

**Qué hace:** ejecuta `l2p-s0` en
`results/campaign-smoke/l2p-s0/`, captura stdout/stderr íntegros, el pico CUDA
del proceso, las métricas persistidas y un checkpoint de prueba para estimar
el presupuesto de disco. Puede tardar varios minutos.

```bash
cd /home/amf380/mammoth
export PYTHON_BIN="/opt/environment/bin/python"
export PYTHONPATH="$HOME/.local/mammoth-pydeps${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p results/campaign-acceptance
bash tools/run_campaign.sh --smoke > results/campaign-acceptance/smoke_driver.txt 2>&1
SMOKE_RC=$?
echo "smoke_exit_code=$SMOKE_RC"
cat results/campaign-acceptance/smoke_driver.txt
test -f results/campaign-smoke/l2p-s0/.done && echo "SMOKE_DONE=si" || echo "SMOKE_DONE=no"
tail -n 100 results/campaign-smoke/l2p-s0/l2p-s0.log
cat results/campaign-smoke/manifest.tsv
```

**Cómo saber que fue bien:** aparece `smoke_exit_code=0` y `SMOKE_DONE=si`;
el manifest termina con exit code `0`; el log muestra construcción del modelo,
datos, iteraciones, evaluación y escritura de resultados. El orquestador solo
crea `.done` si también existen el JSON de picos y un registro nuevo de
`logs.pyd` que supera el gate de persistencia.

Si dice que saltó un `.done` anterior, comprueba en el JSON del subpaso 2.5 que
`git_sha` coincide con `git rev-parse HEAD`. Si no coincide, detente y devuelve
ambos SHA.

**Si falla:** devuelve `smoke_driver.txt`, `manifest.tsv` y las últimas 200
líneas del log:

```bash
cat results/campaign-acceptance/smoke_driver.txt
cat results/campaign-smoke/manifest.tsv 2>/dev/null
tail -n 200 results/campaign-smoke/l2p-s0/l2p-s0.log 2>/dev/null
```

### 2.4. Repetir explícitamente el gate de persistencia

**Qué hace:** lee `logs.pyd` con un parser AST seguro y verifica el LR efectivo
de L2P (`0.03 × 64 / 256 = 0.0075`), un `epoch_time` para el smoke y la matriz
triangular class-IL. En una campaña completa exigirá todas las
`accuracy_i_task_j` con `i <= j`, no solo la fila final; Mammoth ya las guarda
de forma nativa, por lo que no se parsea texto del log.

```bash
cd /home/amf380/mammoth
export PYTHON_BIN="/opt/environment/bin/python"
export PYTHONPATH="$HOME/.local/mammoth-pydeps${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" tools/check_persistence.py \
  --logs data/results/class-il/seq-cifar100-224/l2p/logs.pyd \
  --model l2p --dataset seq-cifar100-224 --seed 0 \
  --notes gate-smoke-l2p-s0 \
  --expected-tasks 1 --expected-epochs 1 \
  --expected-lr 0.0075 --expected-batch-size 64 \
  --after-line 0 \
  > results/campaign-acceptance/check_persistence.txt 2>&1
CHECK_RC=$?
echo "check_persistence_exit_code=$CHECK_RC"
cat results/campaign-acceptance/check_persistence.txt
```

**Cómo saber que fue bien:** aparece `check_persistence_exit_code=0`, una
matriz con la celda `T1/T1` y `OK: gate de persistencia superado`.

**Si falla:** devuelve completo `check_persistence.txt` y las últimas dos
líneas (no todo el historial) de `logs.pyd`:

```bash
cat results/campaign-acceptance/check_persistence.txt
tail -n 2 data/results/class-il/seq-cifar100-224/l2p/logs.pyd
```

### 2.5. Inspeccionar pico CUDA, sampler y checkpoint de prueba

**Qué hace:** comprueba las dos mediciones de memoria. `peak.json` es la métrica
primaria, local al proceso PyTorch y aislada del vecino; el CSV documenta la
contención global. También muestra el tamaño real del checkpoint usado para
decidir si caben 15 finales dejando unos 15 GiB de margen.

```bash
cd /home/amf380/mammoth
export PYTHON_BIN="/opt/environment/bin/python"
export PYTHONPATH="$HOME/.local/mammoth-pydeps${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -m json.tool results/campaign-smoke/l2p-s0/peak.json
echo "--- filas recientes del sampler ---"
tail -n 20 results/campaign/gpu_samples.csv
echo "--- filas del PID exacto del smoke ---"
"$PYTHON_BIN" - <<'PY'
import csv
import json
from pathlib import Path

peak = json.loads(Path("results/campaign-smoke/l2p-s0/peak.json").read_text())
pid = str(peak["pid"])
with Path("results/campaign/gpu_samples.csv").open(newline="") as handle:
    rows = [row for row in csv.DictReader(handle) if row["pid"] == pid]
print(f"smoke_pid={pid}; muestras_en_csv={len(rows)}")
for row in rows[-10:]:
    print(row)
PY
echo "--- checkpoint de prueba y espacio ---"
ls -lh results/campaign-smoke/l2p-s0/checkpoints/*.pt
du -ch results/campaign-smoke/l2p-s0/checkpoints/*.pt
df -h "$HOME"
quota -s 2>&1 || true
```

**Cómo saber que fue bien:** el JSON contiene `peak_allocated_mib`,
`peak_reserved_mib`, PID, timestamps, `duration_seconds`, argv y `git_sha`; los
picos son mayores que cero. Debe haber muestras del PID del smoke en el CSV.
Que aparezcan además PIDs ajenos es correcto. Debe existir un `.pt` de prueba.

El wrapper usa `atexit`: una muerte por `SIGKILL` no puede escribir JSON. En
ese caso no habrá `.done` y la ejecución se repetirá al reanudar.

**Si falla:** devuelve la salida completa de este subpaso, más
`results/campaign/gpu_sampler.log`.

### 2.6. Simular los 15 runs sin entrenar

**Qué hace:** imprime el plan, en el orden metodológico obligatorio
`l2p-s0`, `dualprompt-s0`, `coda_prompt-s0`, después seed 1, etc. También
calcula con el checkpoint de prueba y el espacio disponible si se guardarán
checkpoints finales. No lanza Python ni crea `.done`.

```bash
cd /home/amf380/mammoth
export PYTHON_BIN="/opt/environment/bin/python"
export PYTHONPATH="$HOME/.local/mammoth-pydeps${PYTHONPATH:+:$PYTHONPATH}"
bash tools/run_campaign.sh --dry-run > results/campaign-acceptance/dry_run.txt 2>&1
DRY_RC=$?
echo "dry_run_exit_code=$DRY_RC"
cat results/campaign-acceptance/dry_run.txt
```

**Cómo saber que fue bien:** `dry_run_exit_code=0`; aparecen exactamente 15
runs, agrupados por seed y en el orden L2P → DualPrompt → CODA-Prompt. Cada uno
usa `seq-cifar100-224`, `best`, batch 64 y device 0, sin flags de debug. La
salida explica si el modo automático **habilita** checkpoints finales (espacio
estimado para 15 más margen de 15 GiB) o los **omite**. No cambies esa decisión
manualmente sin revisarla. Si `quota` no existe o su salida no se puede
interpretar con seguridad, la política automática los omite de forma
conservadora: no presupone que todo el espacio mostrado por `df` esté dentro de
tu cuota personal. Hay una excepción comprobada para este servidor: si
`quota -w` no devuelve ningún registro, termina con código 1 y el filesystem
local `ext2`/`ext3`/`ext4` de `$HOME` está montado sin opciones de quota, se
considera que no hay cuota de usuario configurada y se usa el espacio de `df`.
La salida lo identifica como `no_quota_record_and_no_mount_options`.

**Si falla o el orden/flags no coinciden:** devuelve `dry_run.txt` y no lances
la campaña.

Al acabar el paso 2, devuelve estos seis ficheros/salidas para revisar el gate
antes de gastar varios días de GPU:

```bash
cat results/campaign-acceptance/env_snapshot.txt
cat results/campaign-acceptance/smoke_driver.txt
cat results/campaign-acceptance/check_persistence.txt
"$PYTHON_BIN" -m json.tool results/campaign-smoke/l2p-s0/peak.json
tail -n 20 results/campaign/gpu_samples.csv
cat results/campaign-acceptance/dry_run.txt
```

## 3. Lanzar la campaña real con `nohup`

Haz este paso únicamente después de validar el gate. El sampler del paso 2.2
debe seguir vivo. La campaña corre los 15 runs **secuencialmente**, no en
paralelo; un fallo queda en el manifest pero no impide probar los posteriores.
El driver completo exige automáticamente el `.done` y el checkpoint de prueba
del smoke; si falta cualquiera de los dos, aborta antes del primer run y debes
volver al paso 2.3.

**Qué hace:** arranca el driver desconectado de la terminal y guarda toda su
salida. Los logs detallados viven en un directorio por run.

```bash
cd /home/amf380/mammoth
export PYTHON_BIN="/opt/environment/bin/python"
export PYTHONPATH="$HOME/.local/mammoth-pydeps${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p results/campaign
if [ -f results/campaign/driver.pid ] && kill -0 "$(cat results/campaign/driver.pid)" 2>/dev/null; then
  echo "NO SE LANZA OTRO: ya hay un driver activo con PID $(cat results/campaign/driver.pid)"
else
  nohup bash tools/run_campaign.sh \
    >> results/campaign/driver.log 2>&1 < /dev/null &
  NOHUP_PID=$!
  disown "$NOHUP_PID" 2>/dev/null || true
  echo "Proceso nohup iniciado con PID $NOHUP_PID; el driver escribira su propio driver.pid"
fi
sleep 3
if [ -f results/campaign/driver.pid ]; then
  DRIVER_PID="$(cat results/campaign/driver.pid)"
  ps -fp "$DRIVER_PID"
else
  echo "No aparecio driver.pid; revisar driver.log"
fi
tail -n 30 results/campaign/driver.log
```

**Cómo saber que fue bien:** `ps` muestra `bash tools/run_campaign.sh`; el log
indica `l2p-s0` (o el primer run todavía incompleto) y se crea su directorio.
En ese momento ya puedes cerrar la terminal, el navegador y tu ordenador.

**Si falla:** devuelve `driver.log`, `manifest.tsv` si existe y
`ps -fp "$DRIVER_PID"`. No ejecutes el bloque una segunda vez mientras el PID
siga activo.

## 4. Consultar el estado los días siguientes

Abre una Terminal nueva y copia:

```bash
cd /home/amf380/mammoth
export PYTHON_BIN="/opt/environment/bin/python"
export PYTHONPATH="$HOME/.local/mammoth-pydeps${PYTHONPATH:+:$PYTHONPATH}"
echo "=== DRIVER ==="
if [ -f results/campaign/driver.pid ] && kill -0 "$(cat results/campaign/driver.pid)" 2>/dev/null; then
  ps -fp "$(cat results/campaign/driver.pid)"
else
  echo "No hay driver activo (puede haber terminado o haberse caido)."
fi
echo "=== RUN EN CURSO ==="
ps -u "$USER" -o pid,etime,cmd | grep '[t]ools/wrapper_peak.py' || echo "Ningun wrapper activo ahora."
echo "=== COMPLETADOS ==="
DONE_COUNT="$(find results/campaign -mindepth 2 -maxdepth 2 -name .done -type f | wc -l)"
echo "$DONE_COUNT / 15 runs con .done"
echo "=== ULTIMOS INTENTOS DEL MANIFEST ==="
tail -n 8 results/campaign/manifest.tsv 2>/dev/null || true
echo "=== DRIVER LOG ==="
tail -n 40 results/campaign/driver.log 2>/dev/null || true
echo "=== LOG DEL RUN MODIFICADO MAS RECIENTEMENTE ==="
CURRENT_LOG="$(ls -1t results/campaign/*/*.log 2>/dev/null | head -n 1)"
if [ -n "$CURRENT_LOG" ]; then echo "$CURRENT_LOG"; tail -n 30 "$CURRENT_LOG"; fi
echo "=== SAMPLER Y GPU ==="
if [ -f results/campaign/gpu_sampler.pid ]; then ps -fp "$(cat results/campaign/gpu_sampler.pid)"; fi
nvidia-smi
```

**Cómo interpretarlo:** durante un run debe verse un `wrapper_peak.py`; el
contador `.done` crece hasta `15 / 15`; el manifest registra PID, inicio, fin,
exit code y rutas de cada intento. `nvidia-smi` muestra memoria total usada por
todos; los `peak.json` miden solo asignaciones PyTorch del proceso propio.

La campaña terminó correctamente cuando ya no hay driver y hay `15 / 15`.
Si el driver terminó con menos de 15, revisa los exit codes/logs y usa el paso
5: los fallidos carecen de `.done`.

Para seguir el driver en vivo usa lo siguiente. `Ctrl+C` detiene solamente
`tail`, no el entrenamiento:

```bash
tail -f /home/amf380/mammoth/results/campaign/driver.log
```

## 5. Reanudar después de una caída

**Qué hace:** vuelve a recorrer el plan; cada run con `.done` se salta y solo
se repiten los incompletos/fallidos. El manifest conserva todos los intentos.

Primero confirma con el paso 4 que no queda un driver activo. Después copia:

```bash
cd /home/amf380/mammoth
export PYTHON_BIN="/opt/environment/bin/python"
export PYTHONPATH="$HOME/.local/mammoth-pydeps${PYTHONPATH:+:$PYTHONPATH}"
if [ -f results/campaign/driver.pid ] && kill -0 "$(cat results/campaign/driver.pid)" 2>/dev/null; then
  echo "NO REANUDAR: el driver sigue activo con PID $(cat results/campaign/driver.pid)"
else
  nohup bash tools/run_campaign.sh \
    >> results/campaign/driver.log 2>&1 < /dev/null &
  NOHUP_PID=$!
  disown "$NOHUP_PID" 2>/dev/null || true
  echo "Proceso nohup de reanudacion iniciado con PID $NOHUP_PID"
fi
sleep 3
if [ -f results/campaign/driver.pid ]; then ps -fp "$(cat results/campaign/driver.pid)"; fi
tail -n 30 results/campaign/driver.log
```

Si el sampler no sobrevivió, vuelve a ejecutar **solo** el bloque 2.2 antes de
reanudar. Si falla, devuelve `driver.log`, las últimas líneas de
`manifest.tsv` y el log del primer run sin `.done`.

## 6. Parada de emergencia

Usa esto solo si necesitas detener voluntariamente la campaña. Mata únicamente
el driver y sus hijos del usuario actual; no usa `pkill python` y no toca los
procesos de otros usuarios.

```bash
cd /home/amf380/mammoth
if [ -f results/campaign/driver.pid ]; then
  DRIVER_PID="$(cat results/campaign/driver.pid)"
  if kill -0 "$DRIVER_PID" 2>/dev/null; then
    for CHILD_PID in $(pgrep -P "$DRIVER_PID" 2>/dev/null); do
      echo "Enviando TERM al hijo $CHILD_PID"
      kill -TERM "$CHILD_PID" 2>/dev/null || true
    done
    echo "Enviando TERM al driver $DRIVER_PID"
    kill -TERM "$DRIVER_PID" 2>/dev/null || true
  fi
fi
sleep 5
ps -u "$USER" -o pid,etime,cmd | grep -E '[t]ools/(run_campaign.sh|wrapper_peak.py)' || echo "Campana detenida."
```

Para detener también el sampler (normalmente conviene dejarlo hasta acabar):

```bash
cd /home/amf380/mammoth
if [ -f results/campaign/gpu_sampler.pid ]; then
  SAMPLER_PID="$(cat results/campaign/gpu_sampler.pid)"
  kill -TERM "$SAMPLER_PID" 2>/dev/null || true
  echo "TERM enviado al sampler $SAMPLER_PID"
fi
```

El run interrumpido no tendrá `.done` y se repetirá completo al reanudar. No
uses `kill -9` salvo último recurso: `SIGKILL` impide al wrapper escribir el
JSON de pico. Si después de cinco segundos aún aparece algún PID propio, copia
la salida de `ps` y pide revisión antes de forzar la muerte.

## Dónde queda cada evidencia

- `results/campaign/manifest.tsv`: intentos, comandos, PID, tiempos, códigos y
  rutas.
- `results/campaign/<model>-s<seed>/<model>-s<seed>.log`: salida completa del
  entrenamiento.
- `results/campaign/<model>-s<seed>/peak.json`: duración y picos CUDA locales
  al proceso.
- `results/campaign/gpu_samples.csv`: uso observado de todos los procesos GPU,
  incluido el vecino.
- `data/results/class-il/seq-cifar100-224/<model>/logs.pyd`: args, accuracy,
  `epoch_times` y matriz triangular class-IL.
- `.done`: existe solo si entrenamiento, JSON y persistencia terminaron bien.

La planificación inicial es de **4–5 días** para los 15 runs. Tras los
primeros runs completos se recalcula la ETA con sus `epoch_times` reales en la
RTX 3060.

## Ampliar de 5 a 30 seeds

Cuando las seeds 0–4 hayan terminado, la ampliación exacta a 30 seeds usa el
rango inclusivo 5–29. El orden se mantiene igual: para cada seed se ejecutan
L2P, DualPrompt y CODA-Prompt. Primero comprueba el plan:

```bash
cd /home/amf380/mammoth
export PYTHON_BIN="/opt/environment/bin/python"
export PYTHONPATH="$HOME/.local/mammoth-pydeps${PYTHONPATH:+:$PYTHONPATH}"

bash tools/run_campaign.sh --dry-run --seed-start 5 --seed-end 29 \
  > results/campaign/dry_run_s5-s29.txt 2>&1

grep -c '^\[RUN\]' results/campaign/dry_run_s5-s29.txt
grep -E 'Checkpoint policy|save final checkpoints|estimate for 75 runs' \
  results/campaign/dry_run_s5-s29.txt
```

El contador debe ser 75. La política de espacio se guarda separadamente en
`results/campaign/checkpoint_policy_s5-s29.txt`. Cuando ya no exista un driver
activo, lanza la ampliación con:

```bash
nohup bash tools/run_campaign.sh --seed-start 5 --seed-end 29 \
  >> results/campaign/driver.log 2>&1 < /dev/null &

NOHUP_PID=$!
disown "$NOHUP_PID" 2>/dev/null || true
echo "Ampliación lanzada con PID $NOHUP_PID"
```

El mismo comando reanuda la ampliación tras una caída: los runs que ya tengan
`.done` se saltan. Al finalizar debe haber 90 marcadores `.done` en total.

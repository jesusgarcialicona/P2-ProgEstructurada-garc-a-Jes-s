"""
Nombre del Alumno: [Jesús Emmanuel García Licona]
Matrícula: [UX25II075]
Fecha: [25/05/2026]
Examen Segundo Parcial - Programación Estructurada
"""
 
# ==========================================
# 1. IMPORTACIÓN DE BIBLIOTECAS ESTÁNDAR
# ==========================================
import datetime
import math
import random
import statistics
import sys
# ==========================================
# 2. DEFINICIÓN DE CONSTANTES
# ==========================================
MAX_EPOCHS = 10
UMBRAL_ERROR_CRITICO = 0.95
# ==========================================
# 3. FUNCIONES DEFINIDAS POR EL USUARIO
# ==========================================
 
def obtener_info_sistema():
    """
    Usa la biblioteca 'sys' para validar el entorno de ejecución.
    Requisitos: Realizar 3 llamadas distintas a la biblioteca 'sys'.
    """
    # Llamada 1: Mostrar la plataforma del sistema operativo
    plataforma = sys.platform
    print(f"  Plataforma del sistema: {plataforma}")
 
    # Llamada 2: Verificar la versión de Python
    version_python = sys.version
    print(f"  Versión de Python: {version_python}")
 
    # Llamada 3: Mostrar el tamaño máximo de un entero en este sistema
    tam_maximo_int = sys.maxsize
    print(f"  Tamaño máximo de entero (sys.maxsize): {tam_maximo_int}")
 
    # Retornamos la plataforma para usarla en validaciones posteriores si se requiere
    return plataforma, version_python
 
 
def simular_metricas_entrenamiento(cantidad_epochs):
    """
    Usa las bibliotecas 'random' y 'datetime' para simular los datos de entrenamiento.
    Requisitos: 3 llamadas a 'random' y 3 llamadas a 'datetime'.
    """
    # --- Llamadas a datetime ---
    # Llamada 1: Obtener la fecha y hora exacta de inicio
    inicio_simulacion = datetime.datetime.now()
 
    # Llamada 2: Formatear la fecha en español (Día/Mes/Año Hora:Minuto:Segundo)
    fecha_formateada = inicio_simulacion.strftime("%d/%m/%Y %H:%M:%S")
    print(f"  Inicio de simulación: {fecha_formateada}")
 
    # Listas para almacenar métricas por epoch
    lista_loss = []
    lista_latencia = []
    lista_predicciones = []
    lista_reales = []
 
    # Posibles eventos de log
    eventos_log = ["Epoch exitoso", "Gradiente inestable", "Actualización de pesos"]
 
    print(f"\n  {'Epoch':<8} {'Loss':<12} {'Latencia(ms)':<16} {'Evento'}")
    print(f"  {'-'*55}")
 
    for epoch in range(1, cantidad_epochs + 1):
        # --- Llamada random 1: Generar la fluctuación del loss (número decimal aleatorio) ---
        loss = random.uniform(0.05, 1.0)
        lista_loss.append(loss)
 
        # --- Llamada random 2: Simular probabilidad de éxito de la iteración ---
        probabilidad_exito = random.random()
 
        # --- Llamada random 3: Seleccionar un evento de log aleatoriamente ---
        evento = random.choice(eventos_log)
 
        # Generar latencia simulada (ms)
        latencia = random.uniform(50.0, 300.0)
        lista_latencia.append(latencia)
 
        # Generar predicción y valor real para RMSE
        prediccion = random.uniform(0.0, 1.0)
        real = random.uniform(0.0, 1.0)
        lista_predicciones.append(prediccion)
        lista_reales.append(real)
 
        print(f"  {epoch:<8} {loss:<12.4f} {latencia:<16.2f} {evento}  (Prob. éxito: {probabilidad_exito:.2f})")
 
    # --- Llamada 3 a datetime: Calcular la diferencia de tiempo entre inicio y fin ---
    fin_simulacion = datetime.datetime.now()
    duracion = fin_simulacion - inicio_simulacion
    print(f"\n  Fin de simulación: {fin_simulacion.strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"  Duración total de la simulación: {duracion.total_seconds():.4f} segundos")
 
    return lista_loss, lista_latencia, lista_predicciones, lista_reales
 
 
def analizar_rendimiento(lista_loss, lista_latencia):
    """
    Usa la biblioteca 'statistics' para analizar el comportamiento del entrenamiento.
    Requisitos: 3 llamadas distintas a la biblioteca 'statistics'.
    """
    # Llamada 1: Calcular la media (promedio) de los valores de loss
    media_loss = statistics.mean(lista_loss)
    print(f"  Media del loss:              {media_loss:.4f}")
 
    # Llamada 2: Calcular la desviación estándar para medir estabilidad
    # (requiere al menos 2 elementos)
    if len(lista_loss) >= 2:
        desviacion_loss = statistics.stdev(lista_loss)
        print(f"  Desviación estándar del loss: {desviacion_loss:.4f}")
    else:
        desviacion_loss = 0.0
        print(f"  Desviación estándar del loss: No calculable (pocos datos)")
 
    # Llamada 3: Obtener la mediana de la latencia del proceso
    mediana_latencia = statistics.median(lista_latencia)
    print(f"  Mediana de la latencia:      {mediana_latencia:.2f} ms")
 
    return media_loss, desviacion_loss, mediana_latencia
 
 
def calcular_rmse(predicciones, reales):
    """
    Usa la biblioteca 'math' para calcular el Root Mean Squared Error (RMSE).
    Requisitos: 3 llamadas distintas a la biblioteca 'math'.
    """
    n = len(predicciones)
 
    if n == 0:
        print("  No hay datos para calcular RMSE.")
        return 0.0
 
    # Acumular la suma de los errores cuadráticos
    suma_errores_cuadraticos = 0.0
 
    for i in range(n):
        diferencia = predicciones[i] - reales[i]
 
        # Llamada math 1: Elevar la diferencia al cuadrado con math.pow
        error_cuadratico = math.pow(diferencia, 2)
        suma_errores_cuadraticos += error_cuadratico
 
    # Calcular el promedio de los errores cuadráticos
    mse = suma_errores_cuadraticos / n
 
    # Llamada math 2: Calcular la raíz cuadrada del MSE para obtener el RMSE
    rmse = math.sqrt(mse)
 
    # Llamada math 3: Usar math.fabs para asegurar que el RMSE reportado sea positivo
    rmse_final = math.fabs(rmse)
 
    print(f"  RMSE calculado:              {rmse_final:.6f}")
 
    return rmse_final
# ==========================================
# 4. PROGRAMA PRINCIPAL (PUNTO DE ENTRADA)
# ==========================================
if __name__ == "__main__":
    print("=" * 55)
    print("      INICIANDO SIMULADOR DE AGENTES DE IA")
    print("=" * 55)
 
    # --- PASO 1: Validar el entorno del sistema ---
    print("\n[1] INFORMACIÓN DEL SISTEMA:")
    print("-" * 40)
    plataforma, version = obtener_info_sistema()
 
    # --- PASO 2: Simular el entrenamiento ---
    print("\n[2] SIMULACIÓN DE ENTRENAMIENTO:")
    print("-" * 40)
    lista_loss, lista_latencia, predicciones, reales = simular_metricas_entrenamiento(MAX_EPOCHS)
 
    # --- PASO 3: Analizar el rendimiento con statistics ---
    print("\n[3] ANÁLISIS DE RENDIMIENTO:")
    print("-" * 40)
    media_loss, desviacion_loss, mediana_latencia = analizar_rendimiento(lista_loss, lista_latencia)
 
    # --- PASO 4: Calcular el RMSE con math ---
    print("\n[4] CÁLCULO DE ERROR (RMSE):")
    print("-" * 40)
    rmse = calcular_rmse(predicciones, reales)
 
    # --- PASO 5: Reporte final y verificación de métricas críticas ---
    print("\n[5] REPORTE FINAL:")
    print("-" * 40)
    print(f"  Media de loss:      {media_loss:.4f}")
    print(f"  Desv. estándar:     {desviacion_loss:.4f}")
    print(f"  Mediana latencia:   {mediana_latencia:.2f} ms")
    print(f"  RMSE:               {rmse:.6f}")
    print(f"  Umbral crítico:     {UMBRAL_ERROR_CRITICO}")
 
    # Verificar si el loss promedio es crítico (usando condicional en lugar de excepciones)
    if media_loss >= UMBRAL_ERROR_CRITICO:
        print("\n  [!] ALERTA CRÍTICA: El loss promedio supera el umbral crítico.")
        print("  [!] Terminando el programa de forma controlada.")
        # Llamada adicional a sys: sys.exit para salida limpia ante métricas críticas
        sys.exit(1)
    else:
        print("\n  [OK] Las métricas están dentro de los parámetros aceptables.")
        print("  [OK] Entrenamiento completado exitosamente.")
 
    print("\n" + "=" * 55)
    print("      SIMULACIÓN FINALIZADA")
    print("=" * 55)
# ==========================================
# 5. CUESTIONARIO DE ANÁLISIS DE BIBLIOTECAS
# ==========================================
 
"""
CUESTIONARIO DE ANÁLISIS DE BIBLIOTECAS
 
---------------------------------------------------------------------------
Pregunta 1 — Uso de Objetos y Métodos
---------------------------------------------------------------------------
En datetime.datetime.now():
  - La primera parte "datetime" es el MÓDULO (la biblioteca que importamos
    con "import datetime").
  - La segunda parte "datetime" es la CLASE que vive dentro de ese módulo.
    Una clase es una plantilla que agrupa datos y comportamientos relacionados.
  - ".now()" es el MÉTODO de esa clase: una función que pertenece a la clase
    y que, al ejecutarse, devuelve un objeto con la fecha y hora actuales.
 
Relación con biblioteca externa:
  Una biblioteca externa (o estándar) es un conjunto de archivos de código
  ya escritos que podemos reutilizar. Al hacer "import datetime" le decimos
  a Python que cargue ese archivo. Dentro de él existe la clase datetime, y
  esa clase expone el método now(). Es decir, la biblioteca es el contenedor;
  la clase es la plantilla; y el método es la acción que ejecutamos sobre
  esa plantilla.
 
---------------------------------------------------------------------------
Pregunta 2 — Diferenciación Técnica
---------------------------------------------------------------------------
Con "import math" (módulo completo):
  Se carga todo el módulo. Para usar cualquier función debemos anteponer
  el nombre del módulo:
      math.sqrt(9)   math.pow(2, 3)   math.fabs(-5)
 
Con "from math import sqrt" (función específica):
  Solo se importa la función sqrt al espacio de nombres actual. Podemos
  invocarla directamente SIN prefijo:
      sqrt(9)
  Sin embargo, math.pow y math.fabs ya no estarían disponibles a menos
  que también se importen explícitamente.
 
Diferencia clave: "import math" requiere el prefijo "math." siempre;
"from math import sqrt" elimina ese prefijo pero limita lo disponible.
 
---------------------------------------------------------------------------
Pregunta 3 — Flujo y Lógica
---------------------------------------------------------------------------
La conexión entre simulación y RMSE sigue estos pasos:
 
  1. simular_metricas_entrenamiento() genera, por cada epoch, un valor de
     "prediccion" (random.uniform) y un valor "real" (random.uniform).
     Ambos se guardan en las listas "lista_predicciones" y "lista_reales".
 
  2. Esas dos listas se retornan como parte de la tupla de resultados de la
     función de simulación y se capturan en el programa principal:
         lista_loss, lista_latencia, predicciones, reales = simular_metricas...
 
  3. Las listas "predicciones" y "reales" se pasan como argumentos directos
     a calcular_rmse(predicciones, reales).
 
  4. Dentro de calcular_rmse(), se itera sobre ambas listas al mismo tiempo
     usando el índice "i", se calcula la diferencia en cada posición, se
     eleva al cuadrado (math.pow), se acumula, se divide entre n para
     obtener el MSE y finalmente se aplica math.sqrt para obtener el RMSE.
 
  De esta forma, los datos fluyen: simulación → listas → función de cálculo.
 
---------------------------------------------------------------------------
Pregunta 4 — Mapeo de Tipos de Datos
---------------------------------------------------------------------------
Dos tipos de datos complejos (colecciones) utilizados:
 
  1. LISTA (list) — "lista_loss", "lista_latencia", "lista_predicciones",
     "lista_reales":
     Se eligió la lista porque necesitamos almacenar múltiples valores del
     mismo tipo en orden de inserción. Las funciones de "statistics" como
     mean(), stdev() y median() aceptan listas (o cualquier iterable) como
     argumento, lo que hace natural este tipo. Una variable simple solo
     podría guardar UN valor a la vez, perdiendo el historial de todos los
     epochs.
 
  2. TUPLA (tuple) — el retorno de simular_metricas_entrenamiento():
     La función devuelve cuatro listas agrupadas en una tupla implícita
     (Python empaqueta múltiples valores de retorno como tupla). Se eligió
     porque los cuatro elementos (loss, latencia, predicciones, reales) son
     un conjunto fijo de resultados que pertenecen a la misma ejecución y
     no deben modificarse una vez retornados; la tupla refuerza esa
     inmutabilidad conceptual frente a una lista de listas.
 
---------------------------------------------------------------------------
Pregunta 5 — Autoevaluación de Abstracción
---------------------------------------------------------------------------
No fue necesario programar la fórmula matemática de la desviación estándar.
Solo se escribió:
    desviacion_loss = statistics.stdev(lista_loss)
 
La biblioteca "statistics" oculta (abstrae) todos los pasos internos:
calcular la media, restar cada valor, elevar al cuadrado las diferencias,
promediarlas y sacar la raíz cuadrada. Nosotros solo le entregamos la lista
y recibimos el resultado.
 
Relación con el concepto de Abstracción:
  La Abstracción consiste en usar un componente (función, módulo, clase)
  conociendo QUÉ hace sin necesidad de saber CÓMO lo hace internamente.
  Las bibliotecas estándar de Python son el ejemplo más claro: nos proveen
  herramientas listas para usar, elevando nuestro nivel de trabajo y
  permitiéndonos enfocarnos en la lógica del problema (el simulador de
  entrenamiento) en lugar de reinventar operaciones matemáticas básicas.
"""
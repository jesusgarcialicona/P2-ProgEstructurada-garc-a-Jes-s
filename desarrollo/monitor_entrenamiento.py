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
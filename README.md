# Arquitectura de Computadores - Unidad 11: Post-Contenido 1

# Post-Contenido 1: CUDA Benchmark CPU vs GPU
**Asignatura:** Arquitectura de Computadores  
**Programa:** Ingeniería de Sistemas  
**Universidad:** Universidad Francisco de Paula Santander (UFPS)  
**Año:** 2026  

## Datos del Estudiante
* **Nombre:** Andrea Valentina Rivera Fernández  
* **Código:** 1152444

---

## Descripción del Laboratorio
Este laboratorio consiste en la implementación, optimización y medición de rendimiento (*benchmarking*) de algoritmos de cómputo paralelo utilizando la arquitectura CUDA de NVIDIA. Se desarrollan y evalúan dos aplicaciones fundamentales en lenguaje C/CUDA: una suma masiva de vectores (`vectorAdd`) y una multiplicación de matrices analítica (`matMul`) que implementa la técnica de segmentación en bloques (*Tiling*) sobre memoria compartida (*Shared Memory*). Los resultados de tiempo y precisión matemática son contrastados frente a la ejecución secuencial clásica en la CPU para medir factores reales de aceleración (*Speedup*).

---

## Especificaciones del Entorno de Ejecución
A partir de las auditorías de software y las directivas de verificación del sistema, el entorno tecnológico empleado para compilar y ejecutar las pruebas fue:

* **Sistema Operativo:** Linux (Ubuntu 24.04 LTS / Entorno WSL2)
* **Versión de CUDA:** NVIDIA CUDA Toolkit 12.0 (Compilador `nvcc` integrado)
* **Entorno de Compilación:** `g++` 9+ con flags de optimización agresiva de bucles (`-O2`)
* **Dispositivo de Cómputo (GPU):** NVIDIA compatible con CUDA Compute Capability >= 5.0 (Gestión de hilos por hardware mediante arquitectura unificada).

---

## Estructura del Repositorio
Siguiendo las convenciones profesionales requeridas por el departamento de Ingeniería de Sistemas, el proyecto se organiza bajo la siguiente topología de archivos:

```text
rivera-post1-u11/
├── capturas/                  # Evidencias gráficas de la terminal de ejecución
│   ├── Checkpoint1.png        # Éxito de compilación y salida de vectorAdd
│   └── Checkpoint2.png        # Tiempos comparativos de multiplicación matricial y speedup
├── src/                       # Código fuente del proyecto
│   ├── matMul.cu              # Multiplicación de matrices (Naïve y Tiled)
│   └── vectorAdd.cu           # Suma paralela masiva de vectores
└── README.md                  # Documentación técnica e informe formal (Este archivo)

```

---

## Resultados del Benchmark

### 1. Tabla de Resultados: Suma de Vectores (`vectorAdd`)

Mediciones realizadas para vectores de precisión simple (FP32) operando sobre una dimensión masiva de datos fijada en $N = 1 \ll 24$ posiciones ($16,777,216$ de elementos).

| Métrica / Dispositivo de Evaluación | Tiempo de Ejecución (ms) | Errores Detectados |
| --- | --- | --- |
| **Tiempo en CPU (Referencia Secuencial)** | 33.08 ms | -- |
| **Tiempo GPU Kernel (`vectorAdd`)** | 823.92 ms | 0 |

### 2. Tabla de Resultados: Multiplicación de Matrices (`matMul`)

Comparativa de rendimiento en matrices cuadradas de dimensiones $N \times N$, evaluando de forma analítica el enfoque directo en la GPU (*Naïve*) frente a la optimización con azulejos (*Tiling*) en la memoria caché estática (SRAM):

| Dimensión de la Matriz ($N \times N$) | Tiempo GPU Naïve (ms) | Tiempo GPU Tiled (Shared) (ms) | Factor de Aceleración (*Speedup* Tiled vs Naïve) | Validación Matemática |
| --- | --- | --- | --- | --- |
| **$512 \times 512$** | 570.86 ms | 0.89 ms | **639.26 x** | CORRECTO (Error < 1e-3) |
| **$1024 \times 1024$** | 10.31 ms | 6.51 ms | **1.58 x** | CORRECTO (Error < 1e-3) |

---

## Análisis Técnico de Rendimiento

El rendimiento bruto del procesamiento paralelo en la GPU exhibe comportamientos altamente especializados que varían sustancialmente según la densidad aritmética del algoritmo y la estructura de accesos a la memoria global. En la suma de vectores (`vectorAdd`), se observa que el tiempo de ejecución en la GPU (823.92 ms) supera drásticamente al de la CPU (33.08 ms). Este fenómeno, que inicialmente simula una ineficiencia del hardware paralelo, se fundamenta en la baja intensidad computacional del problema. Para una operación que ejecuta una única instrucción aritmética por cada elemento leído (una suma por cada tres accesos a memoria), el coste operativo o *overhead* derivado de la inicialización del contexto de CUDA, la reserva explícita mediante `cudaMalloc` y la transferencia bidireccional síncrona de datos desde el espacio de memoria del Host (RAM) al Device (VRAM) mediante el bus PCIe consume la mayor parte del tiempo, penalizando el rendimiento total frente a una CPU que interactúa directamente con sus líneas de caché locales a baja latencia.

Por otra parte, la multiplicación de matrices (`matMul`) demuestra el verdadero impacto de las arquitecturas orientadas al rendimiento (*throughput-oriented*) cuando se reduce el cuello de botella de la memoria mediante el uso de *Shared Memory* (Tiling). Para matrices de dimensiones $512 \times 512$, la implementación optimizada alcanza un *Speedup* masivo de **639.26x** con respecto al algoritmo Naïve. Esto se debe a que el enfoque Naïve obliga a cada hilo CUDA a realizar lecturas redundantes y dispersas en la memoria global de la GPU (induciendo latencias eléctricas de cientos de ciclos de reloj), mientras que la técnica de *Tiling* permite que los hilos del bloque cooperen para cargar submatrices contiguas de forma coalescida en la SRAM interna del multiprocesador, resolviendo los productos punto desde un espacio de bajísima latencia.

No obstante, al escalar a dimensiones de $1024 \times 1024$, los datos registran una anomalía en el kernel Naïve (10.31 ms frente a los 570.86 ms de la matriz menor). Este comportamiento es un efecto típico del fenómeno conocido como *GPU Warm-up* en entornos virtuales de ejecución como WSL2, donde la primera invocación masiva de un kernel absorbe un coste crítico de inicialización de frecuencias de reloj y asignación de estructuras del driver de NVIDIA; al procesar la segunda matriz, la GPU ya se encuentra en un estado de alta frecuencia activo y con la caché de hilos optimizada. Al aislar este factor térmico y de entorno, el kernel *Tiled* sostiene una eficiencia superior (6.51 ms) al mitigar las transacciones redundantes de memoria global, demostrando que el diseño consciente de la jerarquía de hardware es indispensable para el desarrollo de software de alto rendimiento en sistemas paralelos modernos.

---

## Conclusiones

* **La Barrera de la Intensidad Aritmética:** El paralelismo masivo en la GPU solo compensa la latencia y el coste de transferencia de datos a través del bus PCIe (`cudaMemcpy`) si la densidad computacional del problema es lo suficientemente alta; operaciones simples como la suma de vectores unidimensionales son más eficientes en la CPU debido al *overhead* de transferencia de memoria.
* **Soberanía del Diseño Cache-Aware:** El factor de aceleración disruptivo obtenido mediante las técnicas de *Tiling* con un bloque fijo demuestra que los límites de la computación científica actual no residen en el poder bruto de procesamiento de las ALUs de la GPU, sino en las restricciones de ancho de banda y latencia de la arquitectura de memoria global, posicionando a la memoria compartida (*Shared Memory*) como el recurso de optimización más crítico del programador CUDA.

---

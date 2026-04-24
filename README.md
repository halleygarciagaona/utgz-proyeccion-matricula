# 📊 Proyección de Matrícula Universitaria mediante Aprendizaje Automático
### Universidad Tecnológica de Gutiérrez Zamora — Veracruz, México

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU_USUARIO/utgz-proyeccion-matricula/blob/main/UTGZ_Proyeccion_Matricula_Colab.py)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0009--4017--5976-brightgreen?logo=orcid)](https://orcid.org/0009-0009-4017-5976)

---

## 📌 Descripción

Este repositorio contiene el código fuente del análisis predictivo de matrícula universitaria desarrollado como parte del artículo académico:

> **"Análisis Predictivo de Matrícula Universitaria mediante Aprendizaje Automático: Comparación de Escenarios de Proyección en una IES Tecnológica Pública Mexicana"**  

El sistema implementa cuatro modelos de aprendizaje automático y genera un **ensamble balanceado** para proyectar la matrícula de la UTGZ en el horizonte 2025-2030, utilizando datos del Formato 911 de ANUIES.

---

## 🎯 Objetivos

- Analizar la evolución histórica de la matrícula (2018-2025) considerando el impacto de COVID-19
- Comparar el desempeño predictivo de cuatro modelos de regresión
- Generar proyecciones desagregadas por género (mujeres/hombres) y nivel educativo (LIC/TSU)
- Proporcionar intervalos de confianza del 95% para cada escenario proyectado

---

## 🤖 Modelos Implementados

| Modelo | Descripción | Peso en Ensamble |
|--------|-------------|-----------------|
| **Regresión Lineal OLS** | Tendencia lineal sobre índice ordinal | 15% |
| **Regresión Polinomial (g=2)** | Captura la forma de U de la serie pandémica | 25% |
| **Holt ETS** | Suavización exponencial con componente de tendencia | 35% |
| **Regresión Logarítmica** | Modela estabilización gradual de la matrícula | 25% |

> **Nota metodológica:** El modelo polinomial presenta R²=0.99 pero con riesgo de sobreajuste (n=7). Por ello su peso en el ensamble se limita al 25% y se aplica un cap de 1.5× el máximo histórico en las proyecciones.

---

## 📁 Estructura del Repositorio

```
utgz-proyeccion-matricula/
│
├── 📄 README.md                               ← Este archivo
├── 📄 LICENSE                                 ← Licencia MIT
│
├── 🐍 UTGZ_Proyeccion_Matricula_Colab.py      ← Código principal (Google Colab)
│
├── 📂 datos/
│   └── historico_anuario_2018-2025.xlsx       ← Datos ANUIES Formato 911
│
└── 📂 resultados/
    ├── proyeccion_matricula_multimodelo.png   ← Figura 1: comparación de modelos
    ├── proyeccion_desagregada.png             ← Figura 2: género y nivel educativo
    └── tabla_proyecciones_multimodelo.csv     ← Tabla de proyecciones 2025-2030
```

---

## 📊 Datos Históricos

Fuente: **ANUIES — Estadística de Educación Superior, Formato 911**

| Año académico | Total | Mujeres | Hombres | LIC | TSU | Var. anual |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2018-2019 | 1,999 | 952 | 1,047 | 669 | 1,330 | — |
| 2019-2020 | 1,485 | 720 | 765 | 539 | 946 | −25.7% |
| 2020-2021 | 1,183 | 621 | 562 | 489 | 694 | −20.3% |
| 2021-2022 | 987 | 522 | 465 | 373 | 614 | −16.6% |
| 2022-2023 | 964 | 500 | 464 | 309 | 655 | −2.3% |
| 2023-2024 | 882 | 431 | 451 | 269 | 613 | −8.5% |
| 2024-2025 | **1,078** | 543 | 535 | 297 | 781 | **+22.2%** |

> 🟠 Los años 2020-2021 a 2022-2023 corresponden al período de impacto COVID-19.  
> 🟢 La recuperación iniciada en 2024-2025 sustenta el Escenario de recuperación.

---

## 🔮 Proyecciones 2025-2030 (Ensamble Balanceado)

| Año | Ensamble | IC 95% inferior | IC 95% superior | Var% vs 2024-25 |
|:---:|:---:|:---:|:---:|:---:|
| 2025-2026 | 1,058 | 755 | 1,361 | −1.9% |
| 2026-2027 | 1,183 | 826 | 1,541 | +9.7% |
| 2027-2028 | 1,359 | 953 | 1,765 | +26.1% |
| 2028-2029 | 1,566 | 1,113 | 2,019 | +45.3% |
| 2029-2030 | 1,654 | 1,156 | 2,152 | +53.4% |

---

## 🚀 Cómo Ejecutar

### Opción A — Google Colab (recomendada)

1. Haz clic en el botón **Open in Colab** al inicio de este README
2. Ejecuta las celdas en orden (Celda 1 → Celda 8)
3. Los archivos se descargarán automáticamente al finalizar

### Opción B — Entorno local

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/utgz-proyeccion-matricula.git
cd utgz-proyeccion-matricula

# 2. Instalar dependencias
pip install numpy pandas matplotlib scikit-learn statsmodels openpyxl

# 3. Ejecutar el script
python UTGZ_Proyeccion_Matricula_Colab.py
```

---

## 📦 Dependencias

```python
numpy >= 1.21
pandas >= 1.3
matplotlib >= 3.4
scikit-learn >= 1.0
statsmodels >= 0.13
openpyxl >= 3.0
```

Todas están disponibles por defecto en Google Colab. No se requiere instalación adicional excepto `statsmodels` (incluida en la Celda 1 del script).

---

## 📐 Métricas de Evaluación

Los modelos se evalúan con tres métricas estándar en minería de datos educativa (Shahiri et al., 2015):

- **R²** — Coeficiente de determinación (bondad de ajuste)
- **MAPE** — Error porcentual absoluto medio (interpretabilidad)
- **RMSE** — Raíz del error cuadrático medio (sensibilidad a valores atípicos)

Los **intervalos de confianza del 95%** se calculan mediante el intervalo de predicción estadístico formal (no MAPE proporcional), el cual se ensancha correctamente con el horizonte de proyección.

---

## 📚 Referencias

```bibtex
@article{pedregosa2011,
  title   = {Scikit-learn: Machine learning in Python},
  author  = {Pedregosa, F. and others},
  journal = {Journal of Machine Learning Research},
  volume  = {12},
  pages   = {2825--2830},
  year    = {2011}
}

@article{shahiri2015,
  title   = {A review on predicting student's performance using data mining techniques},
  author  = {Shahiri, A. M. and Husain, W. and Rashid, N. A.},
  journal = {Procedia Computer Science},
  volume  = {72},
  pages   = {414--422},
  year    = {2015},
  doi     = {10.1016/j.procs.2015.12.157}
}

@article{marmolejo2023,
  title   = {Distributional regression modeling via generalized additive models
             for location, scale, and shape},
  author  = {Marmolejo-Ramos, F. and others},
  journal = {WIREs Data Mining and Knowledge Discovery},
  volume  = {13},
  number  = {1},
  pages   = {e1479},
  year    = {2023},
  doi     = {10.1002/widm.1479}
}
```

---

## 👤 Autor

**[Halley Garcia]**  
Docente Investigador — Universidad Tecnológica de Gutiérrez Zamora  
Veracruz, México  

[![ORCID](https://img.shields.io/badge/ORCID-0009--0009--4017--5976-brightgreen?logo=orcid)](https://orcid.org/0009-0009-4017-5976)

---

## 📄 Licencia

Este proyecto está bajo la licencia **MIT**. Puedes usar, modificar y distribuir el código citando la fuente.

```
MIT License — Copyright (c) 2025 [Halley-Garcia]
```

---

## 🔗 Cómo citar este repositorio

Si utilizas este código en tu investigación, por favor cítalo como:

> Garcia-Halley. (2025). *Proyección de Matrícula Universitaria mediante Aprendizaje Automático — UTGZ* [Software]. GitHub. https://github.com/TU_USUARIO/utgz-proyeccion-matricula

---

<div align="center">
  <sub>Desarrollado con Python 🐍 · scikit-learn · statsmodels · matplotlib</sub><br>
  <sub>Universidad Tecnológica de Gutiérrez Zamora · ANUIES Formato 911</sub>
</div>

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║   PROYECCIÓN DE MATRÍCULA — UTGZ, Gutiérrez Zamora, Veracruz               ║
# ║   Modelos: Lineal · Polinomial · Holt ETS · Logarítmico · Ensamble         ║
# ║   Fuente: ANUIES Formato 911 (2018-2025)                                   ║
# ║   Adaptado para Google Colab — ejecutar celda por celda                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  CELDA 1 — Instalación de dependencias                                     │
# │  ▶ Ejecuta esta celda primero. Solo necesaria la primera vez.              │
# └─────────────────────────────────────────────────────────────────────────────┘
# !pip install statsmodels --quiet


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  CELDA 2 — Importaciones                                                   │
# └─────────────────────────────────────────────────────────────────────────────┘
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from IPython.display import display
import warnings
warnings.filterwarnings("ignore")

# Configuración de fuentes y estilo
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120

print("✅ Librerías importadas correctamente")


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  CELDA 3 — Datos históricos                                                │
# │  ✏️  Modifica aquí si tienes datos actualizados                            │
# └─────────────────────────────────────────────────────────────────────────────┘
DATOS = {
    'año':   ['2018-19', '2019-20', '2020-21', '2021-22',
              '2022-23', '2023-24', '2024-25'],
    'total': [1999,      1485,      1183,       987,
               964,       882,      1078],
    'muj':   [ 952,       720,       621,       522,
               500,       431,       543],
    'hom':   [1047,       765,       562,       465,
               464,       451,       535],
    'lic':   [ 669,       539,       489,       373,
               309,       269,       297],
    'tsu':   [1330,       946,       694,       614,
               655,       613,       781],
}

# ── Configuración general ──────────────────────────────────────────────────────
AÑOS_PROY  = ['2025-26', '2026-27', '2027-28', '2028-29', '2029-30']
N_PROY     = len(AÑOS_PROY)

# Pesos del ensamble (deben sumar 1.0)
W_LIN  = 0.15   # Regresión Lineal
W_POLY = 0.25   # Polinomial grado 2  (limitado por sobreajuste)
W_HOLT = 0.35   # Holt ETS           (mayor peso: mejor para series cortas)
W_LOG  = 0.25   # Logarítmico

# Colores de las gráficas
COLORES = {
    'hist':  '#1a6db5',
    'lin':   '#e05c2a',
    'poly':  '#8e44ad',
    'holt':  '#27ae60',
    'log':   '#d4ac0d',
    'ens':   '#2c3e50',
}

# ── Preparación de arrays ──────────────────────────────────────────────────────
df       = pd.DataFrame(DATOS)
n        = len(df)
y        = df['total'].values.astype(float)
X        = np.arange(1, n + 1).reshape(-1, 1)
X_fut    = np.arange(n + 1, n + N_PROY + 1).reshape(-1, 1)
AÑOS_HIST = df['año'].tolist()
XTICKS   = AÑOS_HIST + AÑOS_PROY
x_mean   = float(X.mean())

print("✅ Datos cargados:")
display(df.rename(columns={
    'año': 'Año', 'total': 'Total', 'muj': 'Mujeres',
    'hom': 'Hombres', 'lic': 'LIC', 'tsu': 'TSU'
}))


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  CELDA 4 — Funciones auxiliares                                            │
# └─────────────────────────────────────────────────────────────────────────────┘
def mape_fn(y_true, y_pred):
    """Error porcentual absoluto medio."""
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

def rmse_fn(y_true, y_pred):
    """Raíz del error cuadrático medio."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def adj_r2_fn(r2v, n_obs, k_params):
    """R² ajustado (penaliza parámetros adicionales)."""
    return 1 - (1 - r2v) * (n_obs - 1) / (n_obs - k_params - 1)

def ic_pred_fn(residuos, n_hist, x_fut_idx, x_mean_val):
    """
    Intervalo de predicción estadístico verdadero.
    Se ensancha correctamente con la distancia al período de entrenamiento.
    """
    s   = np.std(residuos, ddof=2)
    Sxx = np.sum((np.arange(1, n_hist + 1) - x_mean_val) ** 2)
    ic  = []
    for xi in x_fut_idx:
        se = s * np.sqrt(1 + 1 / n_hist + (xi - x_mean_val) ** 2 / Sxx)
        ic.append(1.96 * se)
    return np.array(ic)

def ext_fn(base_val, pred_arr):
    """Extiende la serie conectando el último punto histórico con la proyección."""
    idx = [n - 1] + list(range(n, n + N_PROY))
    val = [base_val] + list(pred_arr)
    return idx, val

print("✅ Funciones auxiliares definidas")


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  CELDA 5 — Entrenamiento de los 4 modelos                                  │
# └─────────────────────────────────────────────────────────────────────────────┘

# ── Modelo 1: Regresión Lineal OLS ────────────────────────────────────────────
lin_m       = LinearRegression().fit(X, y)
y_lin_fit   = lin_m.predict(X)
y_lin_pred  = lin_m.predict(X_fut).clip(min=500)    # floor mínimo realista
r2_lin      = round(r2_score(y, y_lin_fit), 3)
mape_lin    = round(mape_fn(y, y_lin_fit), 2)
rmse_lin    = round(rmse_fn(y, y_lin_fit), 1)
ic_lin      = ic_pred_fn(y - y_lin_fit, n, X_fut.flatten(), x_mean)

# ── Modelo 2: Regresión Polinomial grado 2 ────────────────────────────────────
poly_tf     = PolynomialFeatures(degree=2, include_bias=False)
X_p         = poly_tf.fit_transform(X)
X_fp        = poly_tf.transform(X_fut)
poly_m      = LinearRegression().fit(X_p, y)
y_poly_fit  = poly_m.predict(X_p)
cap_poly    = 1.5 * max(y)                           # cap: evita explosión extrapolada
y_poly_pred = poly_m.predict(X_fp).clip(min=500, max=cap_poly)
r2_poly     = round(r2_score(y, y_poly_fit), 3)
mape_poly   = round(mape_fn(y, y_poly_fit), 2)
rmse_poly   = round(rmse_fn(y, y_poly_fit), 1)
ar2_poly    = round(adj_r2_fn(r2_poly, n, 2), 3)
ic_poly     = ic_pred_fn(y - y_poly_fit, n, X_fut.flatten(), x_mean)

# ── Modelo 3: Holt Exponential Smoothing ─────────────────────────────────────
holt_m      = ExponentialSmoothing(
    y, trend='add', initialization_method='estimated'
).fit(optimized=True)
y_holt_fit  = holt_m.fittedvalues
y_holt_pred = holt_m.forecast(N_PROY).clip(min=500)
r2_holt     = round(r2_score(y, y_holt_fit), 3)
mape_holt   = round(mape_fn(y, y_holt_fit), 2)
rmse_holt   = round(rmse_fn(y, y_holt_fit), 1)
alpha_h     = round(holt_m.params['smoothing_level'], 3)
beta_h      = round(holt_m.params['smoothing_trend'], 3)
sigma_h     = np.std(y - y_holt_fit, ddof=1)
ic_holt     = np.array([1.96 * sigma_h * np.sqrt(h) for h in range(1, N_PROY + 1)])

# ── Modelo 4: Regresión Logarítmica ──────────────────────────────────────────
X_log       = np.log(X.flatten()).reshape(-1, 1)
X_log_f     = np.log(X_fut.flatten()).reshape(-1, 1)
log_m       = LinearRegression().fit(X_log, y)
y_log_fit   = log_m.predict(X_log)
y_log_pred  = log_m.predict(X_log_f).clip(min=500)
r2_log      = round(r2_score(y, y_log_fit), 3)
mape_log    = round(mape_fn(y, y_log_fit), 2)
rmse_log    = round(rmse_fn(y, y_log_fit), 1)
ic_log      = ic_pred_fn(y - y_log_fit, n, X_fut.flatten(), x_mean)

# ── Ensamble ponderado ────────────────────────────────────────────────────────
y_ens_pred  = (W_LIN  * y_lin_pred  +
               W_POLY * y_poly_pred +
               W_HOLT * y_holt_pred +
               W_LOG  * y_log_pred)
ic_ens      = (W_LIN  * ic_lin  +
               W_POLY * ic_poly +
               W_HOLT * ic_holt +
               W_LOG  * ic_log)

print("✅ Modelos entrenados")

# ── Tabla de métricas ─────────────────────────────────────────────────────────
metricas = pd.DataFrame({
    'Modelo':         ['Lineal (OLS)', 'Polinomial g2', 'Holt ETS', 'Logarítmico'],
    'R²':             [r2_lin,  r2_poly,  r2_holt,  r2_log],
    'Adj-R²':         [round(adj_r2_fn(r2_lin,  n, 1), 3),
                       ar2_poly,
                       round(adj_r2_fn(r2_holt, n, 2), 3),
                       round(adj_r2_fn(r2_log,  n, 1), 3)],
    'MAPE (%)':       [mape_lin, mape_poly, mape_holt, mape_log],
    'RMSE':           [rmse_lin, rmse_poly, rmse_holt, rmse_log],
    'Peso Ensamble':  [f'{W_LIN:.0%}', f'{W_POLY:.0%}',
                       f'{W_HOLT:.0%}', f'{W_LOG:.0%}'],
})

print("\n📊 MÉTRICAS DE AJUSTE HISTÓRICO:")
display(metricas)
print("\n⚠️  Nota: Polinomial tiene R² alto pero n=7 → riesgo de sobreajuste.")
print("   Por eso su peso en el ensamble se limita al 25%.")

# ── Tabla de proyecciones ─────────────────────────────────────────────────────
proyecciones = pd.DataFrame({
    'Año':            AÑOS_PROY,
    'Lineal':         y_lin_pred.astype(int),
    'Polinomial(cap)':y_poly_pred.astype(int),
    'Holt ETS':       y_holt_pred.astype(int),
    'Logarítmico':    y_log_pred.astype(int),
    'Ensamble':       y_ens_pred.astype(int),
    'IC Inf 95%':     np.maximum(0, y_ens_pred - ic_ens).astype(int),
    'IC Sup 95%':     (y_ens_pred + ic_ens).astype(int),
    'Var% vs 24-25':  [f"{((v/1078)-1)*100:+.1f}%" for v in y_ens_pred],
})

print("\n🔮 PROYECCIONES 2025-2030:")
display(proyecciones)


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  CELDA 6 — FIGURA 1: Comparación de todos los modelos                      │
# └─────────────────────────────────────────────────────────────────────────────┘
fig = plt.figure(figsize=(20, 13))
fig.patch.set_facecolor('#f5f6fa')
gs  = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

# ══ Panel superior — comparación completa ══════════════════════════════════════
ax0 = fig.add_subplot(gs[0, :])
ax0.set_facecolor('white')

# Área de impacto COVID
ax0.axvspan(1.5, 4.5, alpha=0.07, color='darkorange')
ax0.text(3.0, 1960, 'Impacto\nCOVID-19', ha='center', fontsize=9,
         color='darkorange', style='italic', fontweight='bold')

# Línea separadora histórico / proyección
ax0.axvline(x=n - 1.5, color='gray', ls=':', lw=1.5, alpha=0.8)
ax0.text(n - 1.38, 1100, '← Histórico   Proyección →',
         fontsize=9, color='dimgray')

# Datos históricos reales
ax0.plot(range(n), y, 'o-', color=COLORES['hist'], lw=2.8,
         markersize=9, zorder=6, label='Matrícula real (ANUIES)')
for xi, yi in zip(range(n), y):
    ax0.annotate(f'{int(yi):,}', (xi, yi), xytext=(0, 12),
                 textcoords='offset points', ha='center',
                 fontsize=8.5, fontweight='bold', color=COLORES['hist'])

# Proyecciones de cada modelo (con punto de continuación desde último histórico)
xi_e, xv_lin  = ext_fn(y[-1], y_lin_pred)
xi_e, xv_poly = ext_fn(y[-1], y_poly_pred)
xi_e, xv_holt = ext_fn(y[-1], y_holt_pred)
xi_e, xv_log  = ext_fn(y[-1], y_log_pred)
xi_e, xv_ens  = ext_fn(y[-1], y_ens_pred)
xi_ic         = [0] + list(ic_ens)

ax0.plot(xi_e, xv_lin,  '--s', color=COLORES['lin'],  lw=1.7, ms=5, alpha=0.75,
         label=f'Lineal (MAPE={mape_lin}%)')
ax0.plot(xi_e, xv_poly, '--^', color=COLORES['poly'], lw=1.7, ms=5, alpha=0.75,
         label=f'Polinomial g2 (MAPE={mape_poly}%, cap 1.5×máx)')
ax0.plot(xi_e, xv_holt, '--D', color=COLORES['holt'], lw=1.7, ms=5, alpha=0.75,
         label=f'Holt ETS (MAPE={mape_holt}%,  α={alpha_h})')
ax0.plot(xi_e, xv_log,  '--P', color=COLORES['log'],  lw=1.7, ms=5, alpha=0.75,
         label=f'Logarítmico (MAPE={mape_log}%)')
ax0.plot(xi_e, xv_ens,  '-o',  color=COLORES['ens'],  lw=2.8, ms=8, zorder=5,
         label='Ensamble balanceado ← recomendado')

# Banda de IC 95% del ensamble
ens_arr = np.array(xv_ens)
ic_arr  = np.array(xi_ic)
ax0.fill_between(xi_e, ens_arr - ic_arr, ens_arr + ic_arr,
                 alpha=0.12, color=COLORES['ens'])

# Etiquetas ensamble
for xi, yi in zip(range(n, n + N_PROY), y_ens_pred):
    ax0.annotate(f'{int(yi):,}', (xi, yi), xytext=(0, -17),
                 textcoords='offset points', ha='center',
                 fontsize=9, fontweight='bold', color=COLORES['ens'])

ax0.set_title(
    'Análisis de Proyección de Matrícula — UTGZ, Gutiérrez Zamora, Veracruz\n'
    'Comparación de 4 modelos + ensamble balanceado  |  Fuente: ANUIES Formato 911 (2018-2025)',
    fontsize=12, fontweight='bold'
)
ax0.set_xticks(range(len(XTICKS)))
ax0.set_xticklabels(XTICKS, rotation=28, ha='right', fontsize=9.5)
ax0.set_ylabel('Estudiantes', fontsize=10)
ax0.set_xlim(-0.4, len(XTICKS) - 0.6)
ax0.set_ylim(400, 2200)
ax0.legend(loc='upper right', fontsize=8.5, framealpha=0.92, ncol=2)
ax0.grid(True, alpha=0.2, ls='--')

# ══ Panel inferior izquierdo — Holt ETS ════════════════════════════════════════
ax1 = fig.add_subplot(gs[1, 0])
ax1.set_facecolor('white')
ax1.set_title(
    f'Holt Exponential Smoothing\nR²={r2_holt}  MAPE={mape_holt}%  α={alpha_h}  β={beta_h}',
    fontsize=10.5, fontweight='bold'
)
ax1.plot(range(n), y,          'o-',  color=COLORES['hist'], lw=2,   ms=6, label='Real')
ax1.plot(range(n), y_holt_fit, 's--', color=COLORES['holt'], lw=1.4, ms=4, alpha=0.7,
         label='Ajuste histórico')
xi_h, xv_h = ext_fn(y[-1], y_holt_pred)
ic_h_ext   = [0] + list(ic_holt)
ax1.plot(xi_h, xv_h, 'D-', color=COLORES['holt'], lw=2.2, ms=6, label='Pronóstico')
arr_h = np.array(xv_h); ic_hv = np.array(ic_h_ext)
ax1.fill_between(xi_h, arr_h - ic_hv, arr_h + ic_hv, alpha=0.15, color=COLORES['holt'])
for xi, yi in zip(range(n, n + N_PROY), y_holt_pred):
    ax1.annotate(f'{int(yi):,}', (xi, yi), xytext=(0, 9),
                 textcoords='offset points', ha='center', fontsize=8, color=COLORES['holt'])
ax1.axvline(x=n - 1.5, color='gray', ls=':', lw=1)
ax1.set_xticks(range(len(XTICKS)))
ax1.set_xticklabels(XTICKS, rotation=30, ha='right', fontsize=7.5)
ax1.set_ylabel('Estudiantes'); ax1.grid(True, alpha=0.2); ax1.legend(fontsize=8)
ax1.set_xlim(-0.4, len(XTICKS) - 0.6)

# ══ Panel inferior centro — Logarítmico ════════════════════════════════════════
ax2 = fig.add_subplot(gs[1, 1])
ax2.set_facecolor('white')
ax2.set_title(f'Regresión Logarítmica\nR²={r2_log}  MAPE={mape_log}%',
              fontsize=10.5, fontweight='bold')
ax2.plot(range(n), y,        'o-',  color=COLORES['hist'], lw=2,   ms=6, label='Real')
ax2.plot(range(n), y_log_fit,'s--', color=COLORES['log'],  lw=1.4, ms=4, alpha=0.7,
         label='Ajuste histórico')
xi_l, xv_l = ext_fn(y[-1], y_log_pred)
ic_l_ext   = [0] + list(ic_log)
ax2.plot(xi_l, xv_l, 'D-', color=COLORES['log'], lw=2.2, ms=6, label='Pronóstico')
arr_l = np.array(xv_l); ic_lv = np.array(ic_l_ext)
ax2.fill_between(xi_l, arr_l - ic_lv, arr_l + ic_lv, alpha=0.15, color=COLORES['log'])
for xi, yi in zip(range(n, n + N_PROY), y_log_pred):
    ax2.annotate(f'{int(yi):,}', (xi, yi), xytext=(0, 9),
                 textcoords='offset points', ha='center', fontsize=8, color=COLORES['log'])
ax2.axvline(x=n - 1.5, color='gray', ls=':', lw=1)
ax2.set_xticks(range(len(XTICKS)))
ax2.set_xticklabels(XTICKS, rotation=30, ha='right', fontsize=7.5)
ax2.set_ylabel('Estudiantes'); ax2.grid(True, alpha=0.2); ax2.legend(fontsize=8)
ax2.set_xlim(-0.4, len(XTICKS) - 0.6)

# ══ Panel inferior derecho — Ensamble ══════════════════════════════════════════
ax3 = fig.add_subplot(gs[1, 2])
ax3.set_facecolor('white')
ax3.set_title(
    f'Ensamble Balanceado\nLin={W_LIN:.0%}  Poly={W_POLY:.0%}  Holt={W_HOLT:.0%}  Log={W_LOG:.0%}',
    fontsize=10.5, fontweight='bold'
)
ax3.plot(range(n), y, 'o-', color=COLORES['hist'], lw=2, ms=7, label='Real', zorder=5)
xi_e2, xv_e2 = ext_fn(y[-1], y_ens_pred)
ic_e2 = [0] + list(ic_ens)
ax3.plot(xi_e2, xv_e2, 'o-', color=COLORES['ens'], lw=2.5, ms=7, label='Ensamble')
arr_e2 = np.array(xv_e2); ic_ev2 = np.array(ic_e2)
ax3.fill_between(xi_e2, arr_e2 - ic_ev2, arr_e2 + ic_ev2,
                 alpha=0.13, color=COLORES['ens'], label='IC 95%')
for xi, yi in zip(range(n, n + N_PROY), y_ens_pred):
    ax3.annotate(f'{int(yi):,}', (xi, yi), xytext=(0, 10),
                 textcoords='offset points', ha='center', fontsize=8.5,
                 fontweight='bold', color=COLORES['ens'])
ax3.axvline(x=n - 1.5, color='gray', ls=':', lw=1)
ax3.axvspan(1.5, 4.5, alpha=0.05, color='darkorange')
ax3.set_xticks(range(len(XTICKS)))
ax3.set_xticklabels(XTICKS, rotation=30, ha='right', fontsize=7.5)
ax3.set_ylabel('Estudiantes'); ax3.grid(True, alpha=0.2); ax3.legend(fontsize=8.5)
ax3.set_xlim(-0.4, len(XTICKS) - 0.6)

plt.savefig('proyeccion_matricula_multimodelo.png', dpi=180, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("✅ Figura 1 guardada como: proyeccion_matricula_multimodelo.png")


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  CELDA 7 — FIGURA 2: Proyecciones desagregadas (género y nivel)            │
# └─────────────────────────────────────────────────────────────────────────────┘
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 11))
fig2.patch.set_facecolor('#f5f6fa')
fig2.suptitle(
    'Proyecciones Desagregadas por Género y Nivel Educativo\n'
    'UTGZ, Gutiérrez Zamora  |  Ensamble: Holt ETS 55% + Polinomial 45%\n'
    'Fuente: ANUIES Formato 911 (2018-2025)',
    fontsize=12, fontweight='bold', y=1.02
)

VARIABLES = [
    ('muj', 'Matrícula Mujeres',      '#c0392b'),
    ('hom', 'Matrícula Hombres',      '#2980b9'),
    ('lic', 'Licenciatura (LIC)',      '#8e44ad'),
    ('tsu', 'Técnico Superior (TSU)', '#16a085'),
]

desg_results = {}

for ax, (col, titulo, color) in zip(axes2.flat, VARIABLES):
    yv = df[col].values.astype(float)

    # Holt ETS por subvariable
    hm       = ExponentialSmoothing(yv, trend='add',
                                     initialization_method='estimated').fit(optimized=True)
    y_fit_v  = hm.fittedvalues
    y_holt_v = hm.forecast(N_PROY).clip(min=100)

    # Polinomial por subvariable (con cap)
    pm       = LinearRegression().fit(poly_tf.fit_transform(X), yv)
    cap_v    = 1.5 * max(yv)
    y_poly_v = pm.predict(poly_tf.transform(X_fut)).clip(min=100, max=cap_v)

    # Ensamble local: 55% Holt + 45% Polinomial
    y_ens_v        = 0.55 * y_holt_v + 0.45 * y_poly_v
    desg_results[col] = y_ens_v

    sigma_v  = np.std(yv - y_fit_v, ddof=1)
    ic_v     = np.array([1.96 * sigma_v * np.sqrt(h) for h in range(1, N_PROY + 1)])
    r2_v     = round(r2_score(yv, y_fit_v), 3)
    mape_v   = round(mape_fn(yv, y_fit_v), 2)

    ax.set_facecolor('white')
    ax.set_title(f'{titulo}\nHolt ETS  R²={r2_v}  MAPE={mape_v}%',
                 fontsize=11, fontweight='bold', color=color)

    ax.plot(range(n), yv,      'o-',  color=color, lw=2.2, ms=7, label='Real')
    ax.plot(range(n), y_fit_v, 's--', color=color, lw=1.3, ms=4, alpha=0.6,
            label='Ajuste Holt')

    xi_v, xv_v = ext_fn(yv[-1], y_ens_v)
    ic_v_ext   = [0] + list(ic_v)
    ax.plot(xi_v, xv_v, 'D-', color=color, lw=2.2, ms=6, label='Proyección ensamble')
    arr_v = np.array(xv_v); ic_vv = np.array(ic_v_ext)
    ax.fill_between(xi_v, arr_v - ic_vv, arr_v + ic_vv, alpha=0.14, color=color)

    for xi, yi in zip(range(n, n + N_PROY), y_ens_v):
        ax.annotate(f'{int(yi):,}', (xi, yi), xytext=(0, 9),
                    textcoords='offset points', ha='center', fontsize=8, color=color)

    ax.axvline(x=n - 1.5, color='gray', ls=':', lw=1)
    ax.axvspan(1.5, 4.5, alpha=0.05, color='darkorange')
    ax.set_xticks(range(len(XTICKS)))
    ax.set_xticklabels(XTICKS, rotation=30, ha='right', fontsize=7.5)
    ax.set_ylabel('Estudiantes'); ax.grid(True, alpha=0.2); ax.legend(fontsize=8)
    ax.set_xlim(-0.4, len(XTICKS) - 0.6)

plt.tight_layout()
plt.savefig('proyeccion_desagregada.png', dpi=180, bbox_inches='tight',
            facecolor=fig2.get_facecolor())
plt.show()
print("✅ Figura 2 guardada como: proyeccion_desagregada.png")


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  CELDA 8 — Exportar resultados a CSV                                       │
# └─────────────────────────────────────────────────────────────────────────────┘
tabla_final = pd.DataFrame({
    'Año':              AÑOS_PROY,
    'Lineal':           y_lin_pred.astype(int),
    'Polinomial_cap':   y_poly_pred.astype(int),
    'Holt_ETS':         y_holt_pred.astype(int),
    'Logaritmico':      y_log_pred.astype(int),
    'Ensamble':         y_ens_pred.astype(int),
    'IC_Inf_95':        np.maximum(0, y_ens_pred - ic_ens).astype(int),
    'IC_Sup_95':        (y_ens_pred + ic_ens).astype(int),
    'Mujeres_proy':     desg_results['muj'].astype(int),
    'Hombres_proy':     desg_results['hom'].astype(int),
    'LIC_proy':         desg_results['lic'].astype(int),
    'TSU_proy':         desg_results['tsu'].astype(int),
    'Var_pct_vs_2425':  [f"{((v/1078)-1)*100:+.1f}%" for v in y_ens_pred],
})

tabla_final.to_csv('tabla_proyecciones_multimodelo.csv', index=False, encoding='utf-8-sig')

print("✅ CSV guardado como: tabla_proyecciones_multimodelo.csv")
print("\n📊 TABLA FINAL — ENSAMBLE BALANCEADO:")
display(tabla_final)

# ── Descarga automática en Colab ──────────────────────────────────────────────
try:
    from google.colab import files
    files.download('proyeccion_matricula_multimodelo.png')
    files.download('proyeccion_desagregada.png')
    files.download('tabla_proyecciones_multimodelo.csv')
    print("\n✅ Archivos descargados automáticamente.")
except ImportError:
    print("\nℹ️  Ejecuta en Google Colab para descarga automática.")
    print("   Los archivos están guardados en el directorio actual.")

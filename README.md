# PFC: Mejoras para primitivas neurales graficas

Repositorio de codigo para el Proyecto de Final de Carrera I (UCSP)

## Tecnicas implementadas

### 1. Soft Mining (Kheradmand et al., CVPR 2024)

Aceleracion del entrenamiento de NeRF mediante muestreo de importancia basado
en Langevin Monte Carlo (LMC). Las particulas de LMC se desplazan hacia
regiones con alto error de reconstruccion, enfocando el entrenamiento en las
zonas mas dificiles de aprender.

**Directorio:** `nf_soft_mining/`
**Script principal:** `nf_soft_mining/train_ngp_nerf_prop.py`
**Arquitectura:** Instant-NGP (HashGrid + tinycudann)
**Dataset:** LLFF (8 escenas), 20K iteraciones

### 2. Preconditioners (Chng et al., 2024)

Optimizacion con precondicionadores conscientes de curvatura (ESGD/ESGD_Max)
sobre arquitecturas Gaussian NeRF. ESGD_Max utiliza el maximo historico del
producto Hessiano-vector como precondicionador diagonal.

**Directorio:** `garf_preconditioners/`
**Script principal:** `garf_preconditioners/train.py`
**Arquitectura:** Gaussian MLP (8 capas, 256 neuronas, sigma=0.1)
**Dataset:** LLFF (fern), 200K iteraciones

## Resultados principales

| Metodo | Arquitectura | PSNR (fern) | SSIM | LPIPS |
|--------|-------------|-------------|------|-------|
| LMC | Instant-NGP | 25.54 | 0.837 | 0.227 |
| Uniforme | Instant-NGP | 24.20 | 0.794 | 0.251 |
| ESGD_Max | Gaussian MLP | 23.02 | 0.655 | 0.444 |
| Adam | Gaussian MLP | 22.34 | 0.634 | 0.452 |

## Estructura

```
├── garf_preconditioners/   # Implementacion GARF de Preconditioners
│   ├── model/              # Modelos NeRF (nerf_gaussian, garf, etc.)
│   ├── optimizers/         # ESGD, ESGD_Max
│   ├── data/               # Loaders de datasets (LLFF, BLEFF)
│   ├── options/            # Archivos de configuracion YAML
│   └── scripts/            # compare_results.py
├── nf_soft_mining/         # Implementacion de Soft Mining + Preconditioners
│   ├── radiance_fields/    # Modelos (ngp, gaussian_nerf, mlp)
│   ├── datasets/           # LMC sampler + loaders (LLFF, Synthetic)
│   ├── optimizers/         # ESGD, ESGD_Max
│   └── train_*.py          # Scripts de entrenamiento
└── README.md
```

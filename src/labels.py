"""Etiquetas en español para variables GSHS 2013 El Salvador.

Derivadas del diccionario de datos SLV_2013_GSHS_v01.xml (OMS/CDC).
"""

from __future__ import annotations

# Demografía
GSHS_LABELS: dict[str, str] = {
    "Q1": "Edad del estudiante",
    "Q2": "Sexo",
    "Q3": "Grado escolar",
    "Q4": "Estatura (m)",
    "Q5": "Peso (kg)",
    # Alimentación e higiene (QN6–QN14)
    "QN6": "Tuvo hambre la mayor parte del tiempo (30 días)",
    "QN7": "Comió fruta 2+ veces al día (30 días)",
    "QN8": "Comió verduras 3+ veces al día (30 días)",
    "QN9": "Bebió refrescos 1+ veces al día (30 días)",
    "QN10": "Comió comida rápida 3+ días (7 días)",
    "QN11": "Cepilló dientes menos de 1 vez al día (30 días)",
    "QN12": "Nunca/rara vez se lavó las manos antes de comer (30 días)",
    "QN13": "Nunca/rara vez se lavó las manos tras usar el baño (30 días)",
    "QN14": "Nunca/rara vez usó jabón al lavarse las manos (30 días)",
    # Violencia e lesiones (QN15–QN21)
    "QN15": "Fue atacado 1+ veces (12 meses)",
    "QN16": "Estuvo en pelea 1+ veces (12 meses)",
    "QN17": "Resultó gravemente lesionado 1+ veces (12 meses)",
    "QN18": "Fractura como lesión más grave",
    "QN19": "Vehículo motorizado causó la lesión más grave",
    "QN20": "Fue intimidado/bullied 1+ días (30 días)",
    "QN21": "Entre intimidados, golpeado/pateado con más frecuencia",
    # Salud mental (QN22–QN27) — constructo objetivo, no usar como features
    "QN22": "Se sintió solo/a la mayoría del tiempo (12 meses)",
    "QN23": "La preocupación le impidió dormir la mayoría del tiempo (12 meses)",
    "QN24": "Consideró suicidarse (12 meses)",
    "QN25": "Hizo plan de suicidio (12 meses)",
    "QN26": "Intentó suicidarse 1+ veces (12 meses)",
    "QN27": "No tiene amigos cercanos",
    # Alcohol y drogas (QN34–QN40)
    "QN34": "Primera bebida alcohólica antes de los 14 años",
    "QN35": "Bebió alcohol 1+ días (30 días)",
    "QN36": "Entre bebedores, 2+ bebidas al día (30 días)",
    "QN37": "Entre bebedores, obtuvo alcohol de amigos",
    "QN38": "Se emborrachó 1+ veces en la vida",
    "QN39": "Tuvo problemas por alcohol 1+ veces en la vida",
    "QN40": "Primera vez que usó drogas antes de los 14 años",
    # Comportamiento sexual (QN44–QN48)
    "QN44": "Alguna vez tuvo relaciones sexuales",
    "QN45": "Primera relación sexual antes de los 14 años",
    "QN46": "2+ parejas sexuales en la vida",
    "QN47": "Usó condón en la última relación sexual",
    "QN48": "Usó anticonceptivo en la última relación sexual",
    # Actividad física y sedentarismo (QN49–QN52)
    "QN49": "Activo 60+ min/día 5+ de los últimos 7 días",
    "QN50": "Caminó/en bicicleta a la escuela 0 de los últimos 7 días",
    "QN51": "Educación física 3+ días por semana",
    "QN52": "Actividades sentado 3+ horas/día (día habitual)",
    # Factores de protección y supervisión parental (QN53–QN58)
    "QN53": "Faltó a clases sin permiso 1+ de los últimos 30 días",
    "QN54": "Compañeros fueron amables la mayoría del tiempo (30 días)",
    "QN55": "Padres revisan la tarea la mayoría del tiempo (30 días)",
    "QN56": "Padres entienden problemas y preocupaciones (30 días)",
    "QN57": "Padres saben qué hace en tiempo libre (30 días)",
    "QN58": "Padres nunca/rara vez revisan sus cosas",
    # Variables derivadas globales
    "qnowtg": "Sobrepeso (derivada OMS)",
    "qnobeseg": "Obesidad (derivada OMS)",
    "qnunwtg": "Bajo peso (derivada OMS)",
    "qnfrvgg": "Comió 5+ porciones frutas/verduras (30 días)",
    "qnpa7g": "Físicamente activo los 7 días pasados",
    "qnpe5g": "Asistió a educación física 5+ días por semana",
    "qnc1g": "Entre bebedores frecuentes, alguna vez tuvo sexo",
    "qnc2g": "Entre preocupados, actualmente es intimidado",
    "IMC": "Índice de Masa Corporal (kg/m²)",
    "Riesgo_Salud_Mental": "Riesgo grave en salud mental",
}

# Etiquetas de categorías para Q1 (edad)
AGE_LABELS: dict[int, str] = {
    1: "≤11 años",
    2: "12 años",
    3: "13 años",
    4: "14 años",
    5: "15 años",
    6: "≥16 años",
}

# Etiquetas de categorías para Q2 (sexo)
SEX_LABELS: dict[int, str] = {
    1: "Masculino",
    2: "Femenino",
}


def get_label(variable: str) -> str:
    """Devuelve la etiqueta en español de una variable GSHS."""
    return GSHS_LABELS.get(variable, variable)


def get_labels(variables: list[str]) -> dict[str, str]:
    """Devuelve un subconjunto de etiquetas para las variables indicadas."""
    return {var: get_label(var) for var in variables}

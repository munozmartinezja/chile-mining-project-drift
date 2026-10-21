"""Explicit normalization catalogs for source labels."""

CONDICIONES = {
    "BASE": "BASE",
    "PROBABLE": "PROBABLE",
    "POSIBLE": "POSIBLE",
    "POTENCIAL": "POTENCIAL",
    "BASE POSIBLE": "POSIBLE",
    "POSIBLE BASE": "POSIBLE",
}

REGIONES = {
    "ARICA Y PARINACOTA": "Arica y Parinacota",
    "TARAPACA": "Tarapacá",
    "ANTOFAGASTA": "Antofagasta",
    "ATACAMA": "Atacama",
    "COQUIMBO": "Coquimbo",
    "VALPARAISO": "Valparaíso",
    "METROPOLITANA": "Metropolitana",
    "METROPOLITANA DE SANTIAGO": "Metropolitana",
    "METROP": "Metropolitana",
    "RM": "Metropolitana",
    "LIBERTADOR GENERAL BERNARDO OHIGGINS": "O'Higgins",
    "OHIGGINS": "O'Higgins",
    "MAULE": "Maule",
    "NUBLE": "Ñuble",
    "BIOBIO": "Biobío",
    "LA ARAUCANIA": "La Araucanía",
    "ARAUCANIA": "La Araucanía",
    "LOS RIOS": "Los Ríos",
    "LOS LAGOS": "Los Lagos",
    "AYSEN": "Aysén",
    "MAGALLANES": "Magallanes",
    "MAGALLANES Y ANTARTICA CHILENA": "Magallanes",
    "VARIAS": "Varias",
    "INTERREGIONAL": "Varias",
}

SECTORES = {
    "COBRE": "Cobre",
    "ORO": "Oro",
    "HIERRO": "Hierro",
    "MINERIA INDUSTRIAL": "Minería industrial",
    "OTROS": "Otros",
    "POLIMETALICO": "Polimetálico",
    "LITIO": "Litio",
    "Estatal - Cu": "ESTATAL_CU",
    "Estatal - Cu/ Plantas Met": "ESTATAL_CU",
    "Gran Min - Cu": "GRAN_MINERIA_CU",
    "Med Min - Cu": "MEDIANA_MINERIA_CU",
    "Min Industrial": "MINERIA_INDUSTRIAL",
    "Min Ind": "MINERIA_INDUSTRIAL",
    "Plantas Met": "PLANTAS_METALURGICAS",
    "Plantas Metalurg": "PLANTAS_METALURGICAS",
    "Plantas Metalurgicas": "PLANTAS_METALURGICAS",
}

TIPOS_PROYECTO = {
    "BROWNFIELD": "Brownfield",
    "GREENFIELD": "Greenfield",
    "REPOSICION": "Reposición",
    "EXPANSION": "Expansión",
    "NUEVO": "Nuevo",
    "Reposición/ Nuevo": "REPOSICION_NUEVO",
}

ETAPAS = {
    "CONSTRUCCION": "Construcción",
    "EJECUCION": "Ejecución",
    "EJECUCION FACTIBILIDAD": "Ejecución/Factibilidad",
    "Ejecución/ Factibilidad": "EJECUCION_FACTIBILIDAD",
    "FACTIBILIDAD": "Factibilidad",
    "PREFACTIBILIDAD": "Prefactibilidad",
    "PREFACT": "Prefactibilidad",
    "PREFAC": "Prefactibilidad",
    "INGENIERIA CONCEPTUAL": "Ingeniería conceptual",
    "INGENIERIA DE DETALLE": "Ingeniería de detalle",
    "OPERACION": "Operación",
}

OPERADOR_ALIASES = {
    "CODELCO": "CODELCO",
    "CORPORACION NACIONAL DEL COBRE": "CODELCO",
    "MINERA ESCONDIDA": "MINERA ESCONDIDA",
    "BHP BILLITON": "BHP",
}

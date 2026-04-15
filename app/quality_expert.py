"""Expert-rule helpers for quality-related features."""

from __future__ import annotations

from app.config import (
    QUALITY_CATEGORY_RULESETS,
    QUALITY_CODE_BY_LEVEL,
    QUALITY_CONDITION_FACTS,
    QUALITY_FINISH_FACTS,
    QUALITY_LEVEL_BY_CODE,
    QUALITY_MATERIAL_FACTS,
)


class QualityExpertEngine:
    """Converts human selections into model-compatible quality values."""

    @staticmethod
    def default_overall_profile(overall_score: int) -> tuple[str, str, str]:
        if overall_score <= 3:
            return (
                "Madera basica o prefabricado",
                "Acabados basicos",
                "Estado regular",
            )
        if overall_score <= 5:
            return (
                "Concreto y bloque estandar",
                "Acabados estandar",
                "Estado funcional",
            )
        if overall_score <= 7:
            return (
                "Ladrillo o piedra de buena calidad",
                "Acabados buenos",
                "Muy bien mantenida",
            )
        if overall_score <= 8:
            return (
                "Metal estructural con acabados premium",
                "Acabados premium",
                "Muy bien mantenida",
            )
        return (
            "Marmol o piedra de lujo",
            "Acabados de lujo",
            "Estado excelente",
        )

    @staticmethod
    def infer_overall_quality_score(
        material_choice: str,
        finish_choice: str,
        condition_choice: str,
    ) -> tuple[int, list[str]]:
        material_score = QUALITY_MATERIAL_FACTS[material_choice]
        finish_score = QUALITY_FINISH_FACTS[finish_choice]
        condition_score = QUALITY_CONDITION_FACTS[condition_choice]

        raw_score = 1.0 + ((material_score + finish_score + condition_score) - 4.0) * 9.0 / 15.0
        score = int(round(raw_score))

        rules = [
            f"Hecho: material aporta {material_score} puntos.",
            f"Hecho: acabados aportan {finish_score} puntos.",
            f"Hecho: conservacion aporta {condition_score} puntos.",
        ]

        if material_score >= 8 and finish_score >= 4 and condition_score >= 4:
            score = max(score, 9)
            rules.append("Regla premium: materiales y acabados altos elevan la calidad minima a 9.")

        if material_score <= 2 and finish_score <= 2:
            score = min(score, 4)
            rules.append("Regla conservadora: material y acabados basicos limitan la calidad a 4.")

        if condition_score <= 2:
            score = min(score, 5)
            rules.append("Regla de mantenimiento: estado regular limita la calidad a 5.")

        score = max(1, min(score, 10))
        return score, rules

    @staticmethod
    def ruleset(feature: str) -> dict[str, object] | None:
        return QUALITY_CATEGORY_RULESETS.get(feature)

    @staticmethod
    def category_options(feature: str) -> tuple[list[str], list[str], list[str]]:
        ruleset = QualityExpertEngine.ruleset(feature)
        if not ruleset:
            return [], [], []

        material = list(dict(ruleset["material_facts"]).keys())
        finish = list(dict(ruleset["finish_facts"]).keys())
        condition = list(dict(ruleset["condition_facts"]).keys())
        return material, finish, condition

    @staticmethod
    def _best_choice_for_level(
        score_map: dict[str, int],
        level: int,
        avoid_zero: bool = False,
    ) -> str:
        items = list(score_map.items())
        if avoid_zero:
            filtered = [item for item in items if item[1] > 0]
            if filtered:
                items = filtered

        best = min(items, key=lambda item: abs(item[1] - level))
        return str(best[0])

    @staticmethod
    def default_category_profile(feature: str, default_code: str) -> tuple[str, str, str]:
        ruleset = QualityExpertEngine.ruleset(feature)
        if not ruleset:
            return "", "", ""

        none_condition = str(ruleset.get("none_condition", ""))
        if none_condition and str(default_code) == "NoBasement":
            no_key = "No aplica (sin sotano)"
            return no_key, no_key, none_condition

        level = QUALITY_LEVEL_BY_CODE.get(str(default_code), 3)
        material_facts = dict(ruleset["material_facts"])
        finish_facts = dict(ruleset["finish_facts"])
        condition_facts = dict(ruleset["condition_facts"])

        material_default = QualityExpertEngine._best_choice_for_level(material_facts, level, avoid_zero=True)
        finish_default = QualityExpertEngine._best_choice_for_level(finish_facts, level, avoid_zero=True)
        condition_default = QualityExpertEngine._best_choice_for_level(condition_facts, level, avoid_zero=True)
        return material_default, finish_default, condition_default

    @staticmethod
    def infer_quality_code(
        feature: str,
        material_choice: str,
        finish_choice: str,
        condition_choice: str,
    ) -> tuple[str, int, list[str]]:
        ruleset = QualityExpertEngine.ruleset(feature)
        if not ruleset:
            return "TA", 3, ["No hay reglas para esta variable, se usa valor promedio."]

        none_condition = str(ruleset.get("none_condition", ""))
        if none_condition and condition_choice == none_condition:
            return "NoBasement", 0, [
                "Hecho: se indico que no existe sotano.",
                "Regla: el modelo recibe codigo NoBasement.",
            ]

        material_facts = dict(ruleset["material_facts"])
        finish_facts = dict(ruleset["finish_facts"])
        condition_facts = dict(ruleset["condition_facts"])

        material_score = int(material_facts[material_choice])
        finish_score = int(finish_facts[finish_choice])
        condition_score = int(condition_facts[condition_choice])

        raw_level = (material_score + finish_score + condition_score) / 3.0
        level = int(round(raw_level))

        rules = [
            f"Hecho: material aporta nivel {material_score}.",
            f"Hecho: acabados aportan nivel {finish_score}.",
            f"Hecho: estado aporta nivel {condition_score}.",
        ]

        if material_score >= 4 and finish_score >= 4 and condition_score >= 4:
            level = max(level, 5)
            rules.append("Regla premium: todos los factores altos elevan la categoria a Excelente.")

        if material_score <= 2 and finish_score <= 2:
            level = min(level, 3)
            rules.append("Regla conservadora: material y acabados basicos limitan a categoria promedio.")

        if condition_score <= 1:
            level = min(level, 2)
            rules.append("Regla de mantenimiento: estado deteriorado limita a categoria baja.")

        level = max(1, min(level, 5))
        code = QUALITY_CODE_BY_LEVEL[level]
        return code, level, rules

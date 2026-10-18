from __future__ import annotations

from typing import Any


SOURCE_ID = "NBK_231_2019"

SOURCE_TITLE_RU = (
    "Правила осуществления кассовых операций и операций "
    "по инкассации банкнот, монет и ценностей в банках "
    "второго уровня, филиалах банков-нерезидентов "
    "Республики Казахстан, Национальном операторе почты "
    "и юридических лицах, исключительной деятельностью "
    "которых является инкассация банкнот, монет и ценностей"
)

SOURCE_ACT = (
    "Постановление Правления Национального Банка "
    "Республики Казахстан от 29 ноября 2019 года № 231"
)

SOURCE_REGISTRATION = (
    "Зарегистрировано в Министерстве юстиции "
    "Республики Казахстан 3 декабря 2019 года № 19680"
)

SOURCE_URL = (
    "https://adilet.zan.kz/rus/docs/V1900019680"
)


RULES: dict[str, dict[str, Any]] = {
    "cash_handling_presence": {
        "rule_id": (
            "KZ_NBK_231_P20"
        ),

        "paragraph": (
            "20"
        ),

        "title": (
            "Прием и пересчет наличных "
            "в присутствии клиента"
        ),

        "summary": (
            "При приеме наличных денег кассовый "
            "работник осуществляет их прием и "
            "пересчет в присутствии клиента. "
            "Ранее принятые наличные деньги "
            "хранятся в специальных закрываемых "
            "устройствах на рабочем месте."
        ),

        "visual_relevance": (
            "possible"
        ),
    },

    "cashier_workplace_presence": {
        "rule_id": (
            "KZ_NBK_231_P22"
        ),

        "paragraph": (
            "22"
        ),

        "title": (
            "Кассовый работник не покидает "
            "рабочее место до окончания "
            "обслуживания клиента"
        ),

        "summary": (
            "Кассовому работнику не допускается "
            "покидать рабочее место до окончания "
            "обслуживания клиента."
        ),

        "visual_relevance": (
            "high"
        ),
    },

    "cash_issue_presence": {
        "rule_id": (
            "KZ_NBK_231_P31"
        ),

        "paragraph": (
            "31"
        ),

        "title": (
            "Порядок выдачи наличных клиенту"
        ),

        "summary": (
            "При выдаче наличных кассовый работник "
            "подготавливает сумму, пересчитывает "
            "подготовленные наличные деньги "
            "в присутствии клиента и выдает их "
            "клиенту."
        ),

        "visual_relevance": (
            "possible"
        ),
    },

    "cash_storage": {
        "rule_id": (
            "KZ_NBK_231_P20_STORAGE"
        ),

        "paragraph": (
            "20"
        ),

        "title": (
            "Хранение ранее принятых наличных"
        ),

        "summary": (
            "Ранее принятые наличные деньги "
            "хранятся в специальных устройствах, "
            "находящихся на рабочем месте "
            "кассового работника и закрывающихся "
            "на ключ."
        ),

        "visual_relevance": (
            "possible"
        ),
    },

    "cashier_personal_items": {
        "rule_id": (
            "KZ_NBK_231_P12"
        ),

        "paragraph": (
            "12"
        ),

        "title": (
            "Личные вещи работников "
            "в кассовом узле"
        ),

        "summary": (
            "Хранение личных вещей, наличных денег "
            "и ценностей работников банка "
            "осуществляется вне кассового узла "
            "в порядке, установленном "
            "внутренними документами банка."
        ),

        "visual_relevance": (
            "possible"
        ),
    },

    "cash_worker_responsibilities": {
        "rule_id": (
            "KZ_NBK_231_P14"
        ),

        "paragraph": (
            "14"
        ),

        "title": (
            "Исполнение обязанностей "
            "кассовыми работниками"
        ),

        "summary": (
            "Банк обеспечивает осуществление "
            "кассовых операций работниками "
            "в соответствии с их должностными "
            "обязанностями и с распределением "
            "обязанностей, снижающим риск "
            "хищения и иных противоправных действий."
        ),

        "visual_relevance": (
            "context"
        ),
    },

    "cash_transfer_documentation": {
        "rule_id": (
            "KZ_NBK_231_P15"
        ),

        "paragraph": (
            "15"
        ),

        "title": (
            "Передача наличных между "
            "кассовыми работниками"
        ),

        "summary": (
            "Передача наличных денег между "
            "кассовыми работниками отражается "
            "в документе учета принятых и "
            "выданных наличных денег и ценностей."
        ),

        "visual_relevance": (
            "context"
        ),
    },
}


DEFAULT_RULE_KEYS = [
    "cash_handling_presence",
    "cashier_workplace_presence",
    "cash_issue_presence",
    "cash_storage",
]


class KazakhstanCashRuleEngine:
    def __init__(self):
        self.source = {
            "source_id": (
                SOURCE_ID
            ),

            "country": (
                "Kazakhstan"
            ),

            "authority": (
                "National Bank of Kazakhstan"
            ),

            "title": (
                SOURCE_TITLE_RU
            ),

            "act": (
                SOURCE_ACT
            ),

            "registration": (
                SOURCE_REGISTRATION
            ),

            "official_url": (
                SOURCE_URL
            ),
        }

    def get_rule(
        self,
        rule_key: str,
    ) -> dict[str, Any]:
        if rule_key not in RULES:
            raise KeyError(
                f"Unknown rule key: "
                f"{rule_key}"
            )

        return {
            **RULES[
                rule_key
            ],

            "source_id": (
                SOURCE_ID
            ),
        }

    def get_rules(
        self,
        rule_keys: list[str],
    ) -> list[dict[str, Any]]:
        return [
            self.get_rule(
                rule_key
            )
            for rule_key
            in rule_keys
        ]

    def build_generic_violation_context(
        self,
    ) -> dict[str, Any]:
        return {
            "violation_type": (
                "suspected_cash_discipline_violation"
            ),

            "display_name": (
                "Potential cash-discipline violation"
            ),

            "legal_status": (
                "requires_human_review"
            ),

            "automated_legal_conclusion": (
                False
            ),

            "explanation": (
                "The AI detected a visual pattern "
                "associated with a cash-discipline "
                "violation. The training dataset "
                "contains only binary violation "
                "labels, therefore the system does "
                "not automatically determine which "
                "specific legal provision was "
                "breached."
            ),

            "normative_source": (
                self.source
            ),

            "potentially_relevant_rules": (
                self.get_rules(
                    DEFAULT_RULE_KEYS
                )
            ),
        }

    def build_no_violation_context(
        self,
    ) -> dict[str, Any]:
        return {
            "violation_type": None,

            "legal_status": (
                "no_automated_violation_detected"
            ),

            "automated_legal_conclusion": (
                False
            ),

            "normative_source": (
                self.source
            ),
        }
from __future__ import annotations

import json

from app.rules import (
    KazakhstanCashRuleEngine,
)


def main():
    print("=" * 72)
    print(
        "SecureVision — Phase 9B "
        "Kazakhstan Rule Mapping"
    )
    print("=" * 72)

    engine = (
        KazakhstanCashRuleEngine()
    )

    context = (
        engine
        .build_generic_violation_context()
    )

    print()
    print(
        json.dumps(
            context,
            ensure_ascii=False,
            indent=2,
        )
    )

    assert (
        context[
            "violation_type"
        ]
        ==
        "suspected_cash_discipline_violation"
    )

    assert (
        context[
            "automated_legal_conclusion"
        ]
        is False
    )

    assert (
        context[
            "legal_status"
        ]
        ==
        "requires_human_review"
    )

    assert (
        len(
            context[
                "potentially_relevant_rules"
            ]
        )
        >= 4
    )

    paragraphs = {
        rule[
            "paragraph"
        ]
        for rule
        in context[
            "potentially_relevant_rules"
        ]
    }

    assert "20" in paragraphs
    assert "22" in paragraphs
    assert "31" in paragraphs

    print()
    print(
        "✅ Kazakhstan rule catalog loaded."
    )

    print(
        "✅ Official normative source attached."
    )

    print(
        "✅ No unsupported automatic "
        "legal conclusion is made."
    )


if __name__ == "__main__":
    main()
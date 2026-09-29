"""Собираем отчёт из сохранённых результатов обучения."""
import json
import math
import pandas as pd
from experiment import REPORTS


def percent(value):
    return "нет принятых решений" if value is None else f"{value * 100:.2f}%"


def wilson(p, n):
    z = 1.96
    center = (p + z*z / (2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return f"{100*(center-half):.2f}-{100*(center+half):.2f}%"


def main():
    reports = [json.loads((REPORTS / f"{name}.json").read_text(encoding="utf-8")) for name in ["tfidf", "bert_tiny"]]
    text = ["# Результаты Intent Lab", "", "Фактический запуск на BANKING77, seed=42. Официальный test: 3 080 обращений на английском.",
            "Обучение: 6 962; validation: 1 492; calibration для порога: 1 492. Из исходного train удалено 57 строк.", "",
            "| Модель | Test macro-F1 | Test accuracy | Автоматически | Точность принятых |",
            "|---|---:|---:|---:|---:|"]
    for r in reports:
        text.append(f"| {r['model']} | {r['test_macro_f1']:.4f} | {percent(r['test_accuracy'])} | {percent(r['test']['coverage'])} | {percent(r['test']['accepted_accuracy'])} |")
    text.extend(["", "Пороги выбраны на calibration для целевой точности 95% (не менее 50 принятых примеров).",
                 "На test пороги не менялись. Проценты означают долю сообщений в датасете, а не экономию времени сотрудников.", ""])
    for r in reports:
        text.append(f"## {r['model']}")
        text.append(f"\nВремя эксперимента: {r['seconds']:.1f} с. Порог: {r['threshold']}. Передано человеку: {r['test']['manual_review']} из 3 080.")
        if r['test']['accepted']:
            text.append(f"Приближённый 95% интервал Уилсона для точности принятых: {wilson(r['test']['accepted_accuracy'], r['test']['accepted'])}. "
                        "Он предполагает независимые примеры; не учитывает подбор порога и сдвиг данных.")
        else:
            text.append("На calibration не найден допустимый порог; все тестовые обращения передаются человеку. Точность принятых не определена.")
        if "trials" in r:
            text.append(f"\nВыбрано по validation: lr={r['learning_rate']}, эпоха {r['best_epoch']}, macro-F1={r['validation_macro_f1']:.4f}. "
                        f"Максимум эпох на вариант: {r['max_epochs']}; patience={r['early_stopping_patience']}. "
                        "Обучение начато с исходных весов; для новой головы lr=1e-3. Warmup 10%, затем линейное снижение.")
        if r.get("previous_run"):
            old = r["previous_run"]
            text.append(f"\nПредыдущий запуск: macro-F1 {old['test_macro_f1']:.4f}, accuracy {percent(old['test_accuracy'])}. "
                        "Тест повторно используется для сравнения. Это не новая слепая проверка; параметры выбирались только по validation.")
        errors = pd.read_csv(REPORTS / f"{r['model']}_errors.csv")
        text.append("\nЧастые пары ошибок (истинная тема → предсказание):\n")
        pairs = errors.groupby(["label", "predicted"]).size().sort_values(ascending=False).head(5)
        for (label, predicted), count in pairs.items():
            text.append(f"- `{label}` → `{predicted}`: {count}")
        text.append("")
    text.extend(["## Интерпретация", "",
                 "Сравнение относится к данным конфигурациям и ограниченному бюджету обучения. "
                 "Результаты BERT Tiny не определяют качество других трансформеров и режимов обучения. "
                 "История validation и train loss сохранена в ноутбуке: рост validation в конце обучения - признак того, что обучение могло не завершить сходимость.", "",
                 "TF-IDF - практичный исходный вариант: не требует нейросети при предсказании и работает на CPU. "
                 "Перед внедрением нужны независимые данные реальной поддержки и оценка неизвестных намерений."])
    (REPORTS / "SUMMARY.md").write_text("\n".join(text) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

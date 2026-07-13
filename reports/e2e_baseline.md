# End-to-end evaluation — baseline

- **Generated:** 2026-07-12 20:39 UTC
- **Provider:** `fallback(openrouter:meta-llama/llama-3.3-70b-instruct:free,gemini:gemini-2.0-flash,groq:llama-3.3-70b-versatile)`
- **Passages per question (k):** 5
- **Questions:** 20

## Metrics

| Metric | Value |
|---|---|
| Mean faithfulness | 0.980 |
| Mean correctness | 0.905 |
| Refusal rate | 0.050 |
| Uncited-answer rate | 0.000 |

Uncited-answer rate is the cite-or-refuse guardrail's false-positive surface:
the share of *answered* questions whose answer carried no valid citation and
would therefore be converted to a refusal at serving time.

## Per-question

| Example | Faithful | Correct | Refused | Cited |
|---|---|---|---|---|
| q001 | 0.80 | 0.60 | no | yes |
| q002 | 1.00 | 1.00 | no | yes |
| q003 | 1.00 | 1.00 | no | yes |
| q004 | 0.80 | 0.80 | no | yes |
| q005 | 1.00 | 1.00 | no | yes |
| q006 | 1.00 | 1.00 | no | yes |
| q007 | 1.00 | 0.80 | no | yes |
| q008 | 1.00 | 1.00 | no | yes |
| q009 | 1.00 | 1.00 | no | yes |
| q010 | 1.00 | 1.00 | no | yes |
| q010b | 1.00 | 1.00 | no | yes |
| q011 | 1.00 | 1.00 | no | yes |
| q012 | 1.00 | 1.00 | no | yes |
| q013 | 1.00 | 0.90 | no | yes |
| q014 | 1.00 | 1.00 | no | yes |
| q015 | 1.00 | 1.00 | no | yes |
| q016 | 1.00 | 1.00 | no | yes |
| q017 | 1.00 | 1.00 | no | yes |
| q018 | 1.00 | 0.00 | yes | no |
| q019 | 1.00 | 1.00 | no | yes |

# Qwen SSD-Offload und DeepSeek 0731

Stand: 16.09.2026. Laufende Konfiguration in `llama-swap/config.yaml` und `LiteLLM/config.yaml`; vorherige Fassungen liegen als `.bak.*` daneben.

## Qwen3.8-Flash-Next-NVFP4

- Bestehender **RadixArk**-Checkpoint, Revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594`; kein Wechsel zum NVIDIA-Checkpoint und kein zweiter Gewichtsdownload.
- B12X/vLLM liest die vollständige FP8-PLE-Tabelle direkt aus den Safetensors-Dateien auf der NVMe (`VLLM_PLE_TABLE_MEMORY=disk`, io_uring). HashK ist im neuen Profil nicht aktiv.
- Image: `eugr/spark-vllm-b12x@sha256:8e7e062186f841453ef0ec6f713043c5b65447decc3835206685128c18e42262`.
- Kontext 262144, eine aktive Sequenz, Prefill-Chunks 2048, FP8-KV-Cache fest auf 4 GiB. Gemeldete Cache-Kapazität: 305987 Tokens.
- Zunächst eager-Ausführung ohne CUDA-Graphs und ohne MTP-Draft; Stabilität vor zusätzlichem Durchsatz.
- Launcher: `llama-swap/scripts/launch-qwen38-ssd.sh`.
- Cache: `/home/sparky/LLMs/qwen38/ssd-runtime-cache`; Originalgewichte unter `qwen38/flashnext-hf` bleiben zwingend erforderlich.

## DeepSeek-V4-Flash-0731-IQ2XXS-DS4

- `antirez/deepseek-v4-gguf`, Revision `f71f23d552d664e523b422157b2befbf74040380`.
- Datei `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`, 86720111488 Bytes (80.76 GiB).
- SHA256 `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`, nach Download vollständig überprüft. Receipt neben der Modelldatei.
- Alle Experten bleiben erhalten; gemischte aggressive IQ2XXS/Q2K-Quantisierung, wichtige Komponenten Q8. Kein REAP-Pruning.
- Engine `Entrpi/ds4`, Commit `76d51ef82a81b70b78e51a3a6ea11946286de976`, mit `make cuda-spark -j2` für GB10 gebaut.
- Lokales Docker-Image `local/ds4-0731:76d51ef`. Basis und Build-Provenienz unter `ds4-0731-container/`.
- 131072 Kontext, Standardausgabe 8192 Tokens, `--no-spec`, native Speicherreserve `--mem-floor-gb 12`, Disk-KV-Budget 4 GiB. Coalescing ist aus; Prefill-Chunks sind auf 1024 begrenzt, damit die Laufzeit keine zusätzlichen Parallel-Buffersätze reserviert.
- Launcher: `llama-swap/scripts/launch-deepseek-0731.sh`.
- Reproduzierbarer, fortsetzbarer Download: `python3 setup/download-deepseek-0731.py`. Vorhandene verifizierte Datei wird wiederverwendet.

## Schutz und Grenzen

Beide Launcher nutzen `guard-large-model.sh`: mindestens 104 GiB MemAvailable auf drei Messungen, gemeinsamer exklusiver Lock und Prüfung jede Sekunde während der gesamten Laufzeit. Unter 10 GiB wird ausschließlich die über die eigene CID-Datei identifizierte Instanz beendet. Keine fremden Dienste werden automatisch abgeschaltet.

Die 104 GiB sind ein konservatives Startbudget, keine vom Kernel reservierte Menge. Andere manuell gestartete Prozesse können weiter RAM verbrauchen. Die Überwachung reduziert OOM-Risiken, kann einen abrupten Treiberfehler oder einen sehr schnellen Allokationssprung aber nicht sicher verhindern. Die alten manuellen Hostskripte sind nicht Teil dieses gemeinsamen Locks.

Modelle über llama-swap auswählen; große Modelle nacheinander betreiben. Bei Ablehnung zuerst unbenutzte GPU-Dienste entladen. Der ausführliche Grund steht im Modell-Log; llama-swap liefert an den Client derzeit einen generischen HTTP 500.

Logs der neuen Container werden zusätzlich unter `llama-swap/runtime/<container>.log` erhalten. Tests und Messungen: `test-results/ssd-ds4-20260916/`. Guard-Tests ohne GPU: `python3 tests/test_large_model_guard.py`.

## Vergleich und Aufräumen

Der alte lokale DeepSeek-GGUF stammt vom Juni, ist die ältere Preview und enthält im Dateinamen weder 0731 noch imatrix. Die neue Datei ist ein anderes Release und kein bloßer neuer Name.

DeepSeek veröffentlicht für das unquantisierte/native 0731-Release erheblich bessere Agentenwerte als für die Preview (Terminal-Bench 2.1: 82.7 vs. 61.8). Das ist kein gemessener IQ2-Vergleich mit deinem Qwen. Ein belastbarer Sieger bei lokalen Coding-/Agentenaufgaben erfordert dieselben Aufgaben, Prompts, Tool-Umgebung und Erfolgskriterien auf beiden konkreten Laufzeiten. Die kleinen API-Checks hier prüfen Funktion, nicht allgemeine Modellqualität.

Quellen:
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
- https://huggingface.co/antirez/deepseek-v4-gguf
- https://github.com/Entrpi/ds4
- https://github.com/eugr/spark-vllm-docker/blob/main/recipes/qwen3.8-flash-next-nvfp4-solo.yaml

Aufräumvorschlag einschließlich gemeinsam genutzter Draft-Modelle: `MODEL-CLEANUP-PLAN.md`. Keine Benutzer-Modellgewichte ohne eindeutige Freigabe gelöscht.

## Abgeschlossene Live-Prüfung

Qwen: Rechenantwort 391 korrekt; Python-Funktion plausibel; automatischer get_weather-Tool-Aufruf mit Berlin korrekt; Geheimcode aus 80039 Eingabetokens korrekt zurückgegeben. 262144 Kontext konfiguriert, bisher bis 80039 Tokens funktional getestet (keine Behauptung eines vollständigen 262K-Qualitätstests). Zuletzt ca. 22 GiB MemAvailable bei geladenem Modell.

Der erste Qwen-Start mit automatisch dimensioniertem 16.84-GiB-KV-Cache wurde durch den Watchdog an der 10-GiB-Grenze beendet; der explizite 4-GiB-Cache beseitigte dieses Problem. Auch DeepSeeks erster Start mit automatischem Parallel-Batching wurde geschützt abgebrochen.

DS4-Hinweis: Der gepinnte Fork bildet standardmäßig angeforderte high/max-Reasoning-Stufen auf low ab, da er für stark quantisierte Gewichte stabileres tiefes Tool-Calling beobachtet hat. `--reasoning-effort-native` würde die nativen Stufen aktivieren; dieses Profil verwendet vorerst den getesteten Fork-Standard.

DeepSeek 0731: Python-Funktion und automatischer Tool-Aufruf bestanden; Geheimcode aus 80028 Eingabetokens korrekt zurückgegeben (143.35 Sekunden einschließlich Prefill). Kurze Antworten erreichten in diesen Checks rund 15 Tokens/s; das ist kein allgemeiner Leistungsbenchmark. Bei geladenem Modell waren ungefähr 12–13 GiB MemAvailable übrig. 131072 Kontext konfiguriert, bis 80028 Eingabetokens geprüft.

Beide generierten Python-Funktionen bestanden fünf konkrete Eingabetests. Alle vier isolierten Guard-Tests bestanden. Die reale Startverweigerung wurde zusätzlich bei rund 52 GiB verfügbarem RAM geprüft (Exit 75, kein Container gestartet); auch die gemeinsame Startsperre wurde mit laufendem DeepSeek geprüft.

Der alte DeepSeek-Eintrag wurde nach erfolgreichen Tests in llama-swap und LiteLLM durch den 0731-Eintrag ersetzt. Beide Dienste wurden neu gestartet und ihre Modelllisten geprüft, bei LiteLLM mit Authentifizierung. Die alte GGUF-Datei bleibt als Rückfall erhalten. Alle zehn zu behaltenden Profile sind weiterhin vorhanden.

Zum Abschluss ist keines der beiden großen Testmodelle geladen. OCR-/TTS-Dienste laufen wieder; vor dem nächsten Großmodellstart müssen genügend Modelle manuell entladen werden. TTL ist für beide Profile 0 (keine zeitgesteuerte Entladung); bei Bedarf über die Modellverwaltung entladen. Die SSD-Auslagerung spart RAM für PLE, ist keine Garantie für schnellere Modellstarts.

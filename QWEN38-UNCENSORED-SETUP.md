# Qwen3.8-27B-Uncensored-NVFP4-DFlash2

Stand: 17.09.2026. Neues zusätzliches Profil, ursprüngliches Qwen3.8-27B-NVFP4-DFlash2 bleibt erhalten.

## Herkunft und Budget

- Zielgewichte: https://huggingface.co/Vtuber-plan/Qwen3.8-27B-Uncensored-NVFP4
- Gepinnte Revision: `b0f5a99384f9db15e22d028a98d8302284e3748d`.
- Rund 19.2 GiB Dateien; ModelOpt NVFP4, unquantisierter lm_head. Die Bezeichnung „Uncensored“ stammt vom Anbieter; sie ist keine Aussage über allgemeine Coding-Qualität oder eine Garantie für jede Antwort.
- Vorhandener Drafter: `/home/sparky/LLMs/qwen38/z-lab/Qwen3.8-27B-DFlash2`.
- Bestehendes SGLang-Image `lmsysorg/sglang:dev-qwen38-27b-dflash2`, gepinnt auf lokale Image-ID `sha256:60166a99b6610e2b4b3febbeb6411e4cbc9338e996ab6c80d3339f86a7b5b9c3`.
- Persistenter Kernel-Cache unter `qwen38/uncensored-runtime-cache` für Folgestarts.
- Textprofil, Kontext 65536 einschließlich Ausgabe, eine gleichzeitige Anfrage, FP8-KV-Cache und maximal 65536 Cache-Tokens. CUDA-Graph-Batchgröße 1, DFlash-Block 8.
- Thinking standardmäßig aus für kurze Antwortlatenz; pro Anfrage mit `chat_template_kwargs: {"enable_thinking": true}` einschaltbar.
- LiteLLM: max_input_tokens 61440, max_output_tokens 4096 (Angaben zur Aufteilung des gemeinsamen Kontextbudgets).
- Speicherfraktion 0.70 ist ein Laufzeitparameter, keine feste 70%-Reservierung des gesamten Host-RAM. Die expliziten Token- und Parallelitätslimits begrenzen zusätzliche Allokationen.

## Start und Schutz

Launcher: `llama-swap/scripts/launch-qwen38-uncensored-dflash2.sh`.

Der bestehende Guard verlangt hier mindestens 44 GiB MemAvailable auf drei Messungen, überwacht die eigene Instanz jede Sekunde und beendet sie unter 10 GiB Reserve. Der gemeinsame Lock verhindert Parallelbetrieb mit den zwei geschützten Großprofilen. Externe GPU-Prozesse und die anderen alten Profile sind nicht Teil dieses Locks. Der Guard ist keine harte Speicherreservierung und keine absolute Absturzgarantie.

Download: `python3 setup/download-qwen38-uncensored.py`. Pinnt die Revision, lädt in fortsetzbaren Ranges und prüft SHA256 jedes LFS-Artefakts. Der Receipt liegt neben den Gewichten. Kein zweiter Drafter-Download.

## Prüfung

Live-Testskript: `python3 tests/smoke_qwen38_uncensored.py`. Prüft eine Python-Funktion mit fünf Eingaben, automatisches Tool-Calling, Recall über etwa 60K Eingabetokens und Streaming-Durchsatz. Ergebnisse und RAM-Verlauf unter `test-results/qwen38-uncensored-20260917/`.

Status vom 21.09.2026: Download vollständig, alle LFS-Dateien SHA256-verifiziert. Profil in llama-swap aktiv gelistet; Router und LiteLLM wurden am 17.09. neu gestartet. Live-Test noch nicht durchgeführt: ursprünglicher Testaufruf scheiterte vor Ausführung an einem falschen Arbeitsverzeichnis. Bei Wiederaufnahme sind nur rund 10 GiB MemAvailable und 44 GiB belegter Swap vorhanden; der reale Guard verweigert den Teststart korrekt. Das normale Qwen3.8-27B-DFlash2 ist inzwischen geladen. Vor dem Test müssen mindestens 44 GiB MemAvailable frei sein. RAM-Bedarf, Geschwindigkeit und DFlash2-Kompatibilität dieser Kombination sind weiterhin nicht live bestätigt.

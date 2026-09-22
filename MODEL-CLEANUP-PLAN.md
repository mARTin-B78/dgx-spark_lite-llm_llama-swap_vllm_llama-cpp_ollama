# Modell-Aufräumplan – 16.09.2026

> **Ausgeführt am 21.09.2026:** Die freigegebenen Kandidaten wurden gelöscht, mit Ausnahme von **Qwen3.5-27B-Uncensored-DFlash-NVFP4**, das erhalten und aktiv konfiguriert bleibt. Die neun betroffenen Profile wurden in llama-swap/config.yaml, LiteLLM/config.yaml und der vorhandenen llama-swap-Beispielkonfiguration vollständig mit `#` auskommentiert. Backups und Löschmanifest: `runs/model-cleanup-20260921/`.
>
> Entfernt wurden zusätzlich beide alten DSpark-Drafter-Verzeichnisse und die separate alte DSpark-support-GGUF-Datei (5.58 GiB). Der neue DeepSeek-0731-Checkpoint und die benötigten Qwen-DFlash-Drafter bleiben erhalten. Tatsächlich freigegeben: **648.7 GiB**, anschließend etwa **732 GiB SSD frei**. Beide Router wurden neu gestartet; Modelllisten einschließlich authentifizierter LiteLLM-Abfrage geprüft. Das alte DeepSeek-Hostskript meldet bei fehlenden Gewichten/Drafter jetzt einen verständlichen Fehler.
>
> Die folgenden Abschnitte dokumentieren die Planung **vor der Löschung**; ihre damaligen Angaben „noch keine Löschung“ sind historisch.


Noch keine Benutzer-Modellgewichte gelöscht. Größen: belegte Dateisystemblöcke (GiB), keine RAM-Werte. Verzeichnisse können zusätzliche Kopien, Cache oder unvollständige Downloads enthalten. Freigabe ist vor endgültigem Löschen separat zu prüfen.

## Kandidaten aus „Obsolete?“

Diese Modelle sind mögliche Redundanzen zu deiner Behalten-Liste; das ist keine Behauptung, dass sie bei jeder Aufgabe schlechter sind.

| Modell | Belegt (GiB) | Pfad |
|---|---:|---|
| Qwen3-Coder-Next-FP8-Dynamic | 77.7 | `/home/sparky/LLMs/vllm/Alibaba/Qwen3-Coder-Next-FP8-Dynamic` |
| Qwen3-Coder-Next-int4-AutoRound | 40.6 | `/home/sparky/LLMs/vllm/Alibaba/Qwen3-Coder-Next-int4-AutoRound` |
| Qwen3.5-35B-A3B-FP8 | 34.9 | `/home/sparky/LLMs/vllm/Alibaba/Qwen3.5-35B-A3B-FP8` |
| Qwen3.5-27B-Uncensored-DFlash-NVFP4 | 18.4 | `/home/sparky/LLMs/vllm/AEON-7/DFlash-Qwen3.5-27B-Uncensored-NVFP4` |
| Qwen3.5-122B-A10B-heretic-v2-NVFP4 | 71.2 | `/home/sparky/LLMs/vllm/catplusplus/Qwen3.5-122B-A10B-heretic-v2-NVFP4` |
| Qwen3.5-122B-A10B-int4-AutoRound | 143.0 | `/home/sparky/LLMs/vllm/Alibaba/Qwen3.5-122B-A10B-int4-AutoRound` |
| Qwen3.5-122B-A10B-NVFP4 | 58.0 | `/home/sparky/LLMs/vllm/txn545/Qwen3.5-122B-A10B-NVFP4` |
| Qwen3.6-27B-PrismaSCOUT-NVFP4 | 18.8 | `/home/sparky/LLMs/vllm/Alibaba/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm` |
| Qwen3.6-35B-A3B-FP8 | 34.9 | `/home/sparky/LLMs/vllm/Alibaba/Qwen3.6-35B-A3B-FP8` |
| Qwen3.6-35B-A3B-int4-AutoRound | 4.0 | `/home/sparky/LLMs/vllm/Intel/Qwen3.6-35B-A3B-int4-AutoRound` |
| Qwen3.6-35B-A3B-PrismaQuant-4.75bit | 21.3 | `/home/sparky/LLMs/vllm/rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm` |

Summe dieser Verzeichnisse: **522.9 GiB** (noch keine garantierte Freigabe).

## Erhalten und Abhängigkeiten

Alle zehn Einträge deiner Behalten-Liste bleiben erhalten. Qwen3.8-27B-NVFP4-DFlash2 steht außerdem unter Obsolete; Behalten hat Vorrang.

- `vllm/z-lab/Qwen3.5-27B-DFlash` (~4 GiB) behalten: beide AEON-Profile referenzieren diesen Drafter.
- Beide AEON-Profile verwenden denselben Multimodal-NVFP4-MTP-Ordner; nicht doppelt zählen.
- `qwen38/RadixArk/Qwen3.8-27B-NVFP4` und `qwen38/z-lab/Qwen3.8-27B-DFlash2` gehören zusammen.
- `qwen38/flashnext-hf` bleibt für das neue Qwen-SSD-Profil erforderlich. Die SSD-Auslagerung liest gerade aus diesen Originaldateien.
- GPT-OSS-120B-Templates und gemeinsame Chat-Templates erhalten.

## Gesondert prüfen

- Qwen3.5-4B-Q4_K_M: kleiner Fallback kann sinnvoll bleiben; Eintrag verweist auf separaten Dienst.
- Qwen3.6-27B-Uncensored-Q4_K: separaten Dienst und genaue GGUF-Zuordnung vor Löschung prüfen.
- Alter DeepSeek-IQ2XXS-Checkpoint (~81 GiB) wurde nach erfolgreichem 0731-Test in llama-swap und LiteLLM ersetzt. Gewichte vorerst als Rückfall behalten.
- Zusätzlicher Ordner `vllm/sjug/Qwen3.5-122B-A10B-NVFP4-resharded` (~72 GiB) ist nicht das aktuell referenzierte txn545-Profil; als weiteren Kandidaten prüfen.
- Alte Draft-Modelle unter `ollama/DeepSeek/DSpark-Drafter*` werden vom neuen Profil ohne Speculation nicht benötigt, können aber von alten Hostskripten genutzt werden.
- Qwen3-Coder-Next-int4 kann als kleineres Coding-Ausweichmodell sinnvoll sein; FP8 plus int4 parallel aufzubewahren kostet zusammen etwa 119 GiB.

Vor Löschung: aktive Container-Mounts, llama-swap, LiteLLM, llama.cpp/models.ini und Hostskripte prüfen. Keine pauschale Löschung des Hugging-Face-Caches oder von Docker-Images.

Genau zugeordnete kleine Dienste: `Qwen3.5-4B-Q4_K_M` nutzt `ollama/Alibaba/Qwen3.5-4B/Qwen3.5-4B-Q4_K_M.gguf` (2.55 GiB); `Qwen3.6-27B-Uncensored-Q4_K` nutzt tatsächlich `ollama/Abiray/Huihui-Qwen3.6-27B-abliterated/qwen3.6-27b-abliterated-Q4_K_M.gguf` (15.41 GiB). Der separate HauhauCS-Q8-Ordner ist nicht dieser Q4-Dienst. Beide Compose-Dienste sind manuell gestartete Profile.

## Priorisierte Empfehlung – 21.09.2026

Erneut gemessen: SSD 83 GiB verfügbar (98 % belegt); RAM rund 11 GiB MemAvailable, 44 GiB Swap belegt. Die bisherigen Kandidatengrößen sind unverändert.

Erste Löschrunde empfohlen: alle elf Kandidaten der obigen Tabelle **außer Qwen3-Coder-Next-int4-AutoRound** (482.3 GiB), dazu der alte DeepSeek-GGUF (80.76 GiB) und der zusätzliche sjug-122B-resharded-Ordner (71.23 GiB). Insgesamt rund **634 GiB** Dateisystembelegung. Das ist eine Empfehlung zur Reduzierung lokaler Redundanz, keine Qualitätsrangliste.

Der alte DeepSeek ist `ollama/DeepSeek/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf`. Nur das 0731-Profil ist aktuell in llama-swap eingetragen; die erfolgreich getestete 0731-Datei im Unterordner unbedingt behalten. Das alte Hostskript `scripts/start-ds4-deepseek.sh` muss bei Entfernung des alten Modells ebenfalls stillgelegt/aktualisiert werden.

Kein laufender eigener Modellcontainer hat einen der vorgeschlagenen Ordner direkt gemountet. llama-swap und llama.cpp mounten allerdings die übergeordneten Modellverzeichnisse; einige Kandidaten sind noch als startbare Profile eingetragen. Beim tatsächlichen Entfernen deshalb die zugehörigen llama-swap-/LiteLLM-Einträge und Download-/Startskripte mitbereinigen. Noch keine Löschung ausgeführt.

Vorläufig behalten: Qwen3-Coder-Next-int4-AutoRound als eigener Coding-Fallback (~40.6 GiB), kleiner Qwen3.5-4B-Fallback (~2.55 GiB), normales Qwen3.8-27B-NVFP4-DFlash2 und sein Drafter, bis die neue uncensored-Kombination live geprüft ist. Deine explizite Behalten-Liste bleibt vollständig erhalten. Auch die neuen uncensored-Gewichte (~19.2 GiB) bleiben für den ausstehenden Test erhalten.

Optional nach Abschaltung der alten DeepSeek-Hostskripte: DSpark-Drafter (16.64 GiB) und DSpark-Drafter-F16 (10.53 GiB); das neue 0731-Profil verwendet beide nicht. Optional außerdem Qwen3.6-27B-Uncensored-Q4_K (15.41 GiB), falls der separate GGUF-Fallback nicht benötigt wird.

Weitere große Bereiche, noch nicht zur Löschung bewertet: ComfyUI-Modelle ~441 GiB, Whisper-Hub ~102 GiB. Nicht pauschal löschen.

RAM und SSD getrennt behandeln: Löschen ungeladener Gewichte gibt hauptsächlich SSD-Platz frei. Für RAM das aktuell geladene normale Qwen3.8-27B-DFlash2 und bei Bedarf unbenutzte TTS-/OCR-/ComfyUI-Modelle entladen. Dateien eines geladenen Modells zu entfernen ist kein verlässlicher Weg, dessen RAM freizugeben.

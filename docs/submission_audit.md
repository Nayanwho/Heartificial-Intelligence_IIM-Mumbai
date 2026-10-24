# Submission audit — 24 October 2026

## Submission identity

- Project: **Santulan: a research prototype for clinician-led diabetes dose review**.
- Team: **Heartificial Intelligence**.
- College: **Indian Institute of Management Mumbai**.
- Team leader: **Shubham Sunil Patil**; team member: **Adarsh Nayan**.
- Repository/folder: `Heartificial-Intelligence_IIM-Mumbai`.
- GitHub: https://github.com/Nayanwho/Heartificial-Intelligence_IIM-Mumbai

The team confirmed these details during submission preparation. Contact details belong in the Unstop form and are not needed in this public repository.

## Required material

The [README checklist](../README.md#submission-checklist) maps every field in the supplied submission-form screenshot to its document or section. The package includes source code, data, fitted models, results, architecture PDF, 17-slide presentation PDF and PPTX, screenshots, figures and two supporting video clips.

**Final video supplied and linked:** [Santulan demonstration](https://youtu.be/pIliad9_z6Q), **20:41**, published by Shubham Patil. YouTube labels it **Unlisted**. Playback and a working live-replay segment at 4:08 were inspected; the public oEmbed endpoint returned the correct title/channel without credentials. This was a playback/content spot-check, not a full word-for-word review of the narration.

**Duration discrepancy disclosed:** the submission form says 15–20 minutes, while the [challenge page](https://unstop.com/hackathons/crp-digital-twin-challenge-2026-happiest-health-1757873) says a minimum of 20 minutes. The supplied video meets the minimum but exceeds the form’s upper figure by 41 seconds. The team explicitly selected this video as final after being informed of the discrepancy. The README and submission note disclose its actual duration; organizer acceptance of the conflicting duration wording cannot be guaranteed.

The two supporting clips last 2:46.76 and 2:01.08 and remain available alongside the full video.

The live challenge page lists the prototype/code submission deadline as **24 October 2026, 7:00 PM IST**. Unstop submission status is separate from the GitHub package; it must be confirmed by the platform after submission.

## Checks performed

- Architecture: one-page PDF rendered and inspected.
- Presentation: all 17 PDF pages rendered and inspected; PPTX archive integrity checked, with 17 slides and 17 speaker-note parts. Slides are embedded images, with editable source in `presentation.html`.
- README: required sections present, local file links resolve, and the fake YouTube URL was removed.
- Files: largest file is under 10 MB; no GitHub 100 MB single-file blocker. Cache files, OS metadata, virtual environments and common secret-file paths are excluded by `.gitignore`.
- Credential-pattern scan of project text files found no matches. This is a bounded scan, not a security certification.
- Reproducibility: all 110 data-file hashes, 19 code-file hashes and the protocol hash match the saved manifest.
- Dataset licence and official source: verified from figshare metadata; [source comparison](data_provenance.md) documents two substantive mirror differences.
- Tests: **59 passed, 1 skipped** using Python 3.13 on macOS after installing the documented `libomp` dependency. Includes documentation consistency, causal inputs, fold separation, alert scoring, replay parity and dashboard smoke checks. The skipped check targets a private external video-script file; it does not skip application validation. The final video was checked separately as described above.
- Tests used the locally installed packages (including LightGBM 4.7.0, pandas 2.3.1 and Streamlit 1.59.1); the original Linux evaluation environment is recorded separately in `requirements-lock.txt` and the manifest.

This audit checks submission packaging, document consistency and the existing validation suite. It does not independently retrain every model, validate clinical claims, or verify every external background reference.

## Final package status

- The final video URL is present in the README checklist, demo section and public-access section.
- The original public upload was verified anonymously: all 275 GitHub file hashes matched the local commit, and the required PDFs, PPTX, licence, README and supporting clips downloaded successfully.
- The final documentation update is checked with `python3 -m pytest -q tests/test_docs_consistency.py`; no model, data or application code changed in this update.
- The repository link for Unstop is **https://github.com/Nayanwho/Heartificial-Intelligence_IIM-Mumbai**.
- The video duration note must accompany the form submission because its stated range differs from the challenge page.

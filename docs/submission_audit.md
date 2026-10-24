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

**One required item is pending: the final narrated unlisted YouTube video URL.** The supporting clips last 2:46.76 and 2:01.08 and do not satisfy the full-video requirement. The form says 15–20 minutes; the live [challenge page](https://unstop.com/hackathons/crp-digital-twin-challenge-2026-happiest-health-1757873) says a minimum of 20 minutes. A 20-minute video satisfies both. The existing speaker-note timings extend beyond 20 minutes and should be paced/trimmed for that duration.

The live challenge page lists the prototype/code submission deadline as **24 October 2026, 7:00 PM IST**. The team will complete the Unstop form.

## Checks performed

- Architecture: one-page PDF rendered and inspected.
- Presentation: all 17 PDF pages rendered and inspected; PPTX archive integrity checked, with 17 slides and 17 speaker-note parts. Slides are embedded images, with editable source in `presentation.html`.
- README: required sections present, local file links resolve, and the fake YouTube URL was removed.
- Files: largest file is under 10 MB; no GitHub 100 MB single-file blocker. Cache files, OS metadata, virtual environments and common secret-file paths are excluded by `.gitignore`.
- Credential-pattern scan of project text files found no matches. This is a bounded scan, not a security certification.
- Reproducibility: all 110 data-file hashes, 19 code-file hashes and the protocol hash match the saved manifest.
- Dataset licence and official source: verified from figshare metadata; [source comparison](data_provenance.md) documents two substantive mirror differences.
- Tests: **59 passed, 1 skipped** using Python 3.13 on macOS after installing the documented `libomp` dependency. Includes documentation consistency, causal inputs, fold separation, alert scoring, replay parity and dashboard smoke checks. The skipped check targets a private external video-script file; it does not skip application validation. The final video content itself is not yet available to check.
- Tests used the locally installed packages (including LightGBM 4.7.0, pandas 2.3.1 and Streamlit 1.59.1); the original Linux evaluation environment is recorded separately in `requirements-lock.txt` and the manifest.

This audit checks submission packaging, document consistency and the existing validation suite. It does not independently retrain every model, validate clinical claims, or verify every external background reference.

## Final video step

1. Add the supplied YouTube URL under `README.md` → Demo video.
2. Verify playback without signing in, the final duration, and visibility of the working prototype.
3. Replace the pending status in the README checklist and public-access section, then push the update.
4. The team submits the repository URL and required team-leader details on Unstop, and checks the video/public-access boxes only after that verification.

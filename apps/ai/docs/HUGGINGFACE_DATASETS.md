# Hugging Face dataset audit

Snapshot date: 2026-10-17.

The search covered banknotes, cashier activity, retail actions, shoplifting,
surveillance anomaly video and hand-object interaction. The machine-readable
decision log lives in `data/external/huggingface_datasets.json` and can be
validated with:

```bash
.venv/bin/python training/audit_external_datasets.py
```

## Result

No public Hugging Face dataset is both a direct match for bank cash-discipline
violations and clearly licensed for the final ForteBank deliverable.

The two candidates that can be used in a narrow auxiliary role are:

- [7xiang/VAU-Bench](https://huggingface.co/datasets/7xiang/VAU-Bench): MIT
  metadata with anomaly classes, time intervals, key evidence and explanations.
  It is useful for the output schema and taxonomy, not as a cashier-video
  training corpus.
- [RichMiguel/Banknotes-Coins](https://huggingface.co/datasets/RichMiguel/Banknotes-Coins):
  MIT, 1,100 labeled 256x256 images. It can support a generic banknote
  appearance experiment but has no KZT ontology or detection boxes.

The closest visual-domain dataset is
[RetailAction](https://huggingface.co/datasets/standard-cognition/RetailAction),
with 42,000 ceiling-view clips and `take`, `put`, and `touch` actions. Its
custom license requires separate commercial permission under the conditions
described in the repository, so it is blocked by default.

[CCTV-Anomaly](https://huggingface.co/datasets/xiaozhang79/CCTV-Anomaly) is a
large 2026 dataset, but its dataset card explicitly restricts the annotations
to research and non-commercial use. This restriction overrides the more
permissive license tag shown in repository metadata for our usage decision.

Public shoplifting mirrors either state research-only use or provide no clear
license/provenance. They must not enter the training set merely because the
files are downloadable.

## Engineering decision

SecureVision will continue training the decision model on the provided videos.
External data may be used only after it appears in the registry with an
approved scope. The first model improvement is therefore a dense-motion
descriptor tailored to the fixed overhead camera, rather than importing a
domain-mismatched or legally incompatible anomaly corpus.

The first grouped experiment found that motion-only features improved train
OOF specificity from 0.13 for the X3D reference to 0.47 while retaining 0.82
recall. This did not survive full sliding-window validation: all four normal
development videos still produced at least one false peak and timestamp error
increased. The motion backend remains experimental and the production default
therefore stays on the previous X3D classifier until object/zone evidence is
added.

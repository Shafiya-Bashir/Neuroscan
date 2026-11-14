# Contributing to NEUROSCAN

We welcome academic feedback, scientific discussions, and reproducibility evaluations. Please review the following guidelines before submitting issues or pull requests.

## Academic & Research Integrity
- All contributions must adhere to high standards of scientific reproducibility.
- If proposing architectural modifications or new loss functions, benchmark them against the baseline 70/20/10 stratified split with quantitative statistical testing (Precision, Recall, F1-Score, and ROC-AUC).

## Code Style & Formatting
- **Python:** Format code using `black` and adhere to PEP 8 standards.
- **PyTorch:** Explicit tensor shape annotations and device-agnostic execution (`mps`, `cuda`, `cpu`).
- **TypeScript/React:** Adhere to Next.js App Router conventions with Tailwind CSS utility classes.

## Pull Request Workflow
1. Fork the repository and create a descriptive branch: `git checkout -b feat/your-feature-name`.
2. Ensure that unit tests and inference tests pass:
   ```bash
   cd ai_engine
   python3 inference.py --image dataset/test/glioma/glioma_0007.png --model scratch
   ```
3. Ensure the Next.js UI compiles cleanly:
   ```bash
   npm run build
   ```
4. Commit your changes with conventional commit messages (`feat: ...`, `fix: ...`, `docs: ...`).
5. Open a Pull Request referencing your academic rationale.

#!/usr/bin/env python3
"""
Rebuild ARITHMETIC_BLACK_HOLE.tex from the pandoc backup.
All transformations in one pass, with proper UTF-8 handling.
"""
import re

with open('ARITHMETIC_BLACK_HOLE_pandoc.tex', 'r', encoding='utf-8') as f:
    text = f.read()
print(f"Pandoc backup: {len(text)} chars, FFFD={text.count(chr(0xFFFD))}")

# ═══════════════════════════════════════════════════════════════════════
# 1. PREAMBLE ADDITIONS
# ═══════════════════════════════════════════════════════════════════════
preamble_extras = r"""
% ── Additional packages for publication formatting ──
\usepackage{fancyhdr}
\usepackage{lastpage}
\usepackage{needspace}
\usepackage{amsthm,mathtools}
\numberwithin{equation}{section}
\setlength{\extrarowheight}{2pt}
\preto{\section}{\needspace{10\baselineskip}}
\preto{\subsection}{\needspace{4\baselineskip}}
\setcounter{secnumdepth}{-1}
\pagestyle{fancy}
\fancyhead[L]{\small The Arithmetic Black Hole}
\fancyhead[R]{\small Preprint --- April 6, 2026}
\fancyfoot[C]{\thepage\ of \pageref{LastPage}}
"""
text = text.replace(r'\begin{document}', preamble_extras + r'\begin{document}')
print("✓ Preamble extras added")

# ═══════════════════════════════════════════════════════════════════════
# 2. SECTION PROMOTION (subsubsection→subsection, subsection→section)
# ═══════════════════════════════════════════════════════════════════════
text = text.replace(r'\subsubsection{', 'PLACEHOLDER_SUBSEC{')
text = text.replace(r'\subsection{', r'\section{')
text = text.replace('PLACEHOLDER_SUBSEC{', r'\subsection{')
print("✓ Section hierarchy promoted")

# ═══════════════════════════════════════════════════════════════════════
# 3. REMOVE TOC (will re-insert after abstract later)
# ═══════════════════════════════════════════════════════════════════════
text = re.sub(
    r'\{\s*\\setcounter\{tocdepth\}\{3\}\s*\\tableofcontents\s*\}',
    '', text
)
print("✓ TOC removed (will re-insert after abstract)")

# ═══════════════════════════════════════════════════════════════════════
# 4. TITLE BLOCK (centered, with full metadata)
# ═══════════════════════════════════════════════════════════════════════
new_title = r"""\begin{document}

% ── Title Block ──
\begin{center}
{\LARGE\bfseries The Arithmetic Black Hole:\\[0.3em]
Softmax Thermodynamics and the Four Eigenvalue Laws\\[0.3em]
of the Prime Gas}\\[1.5em]

{\large Antonio P.\ Matos}\\[0.5em]
\small ORCID: \href{https://orcid.org/0009-0002-0722-3752}{0009-0002-0722-3752}\\[0.3em]
\small Independent Researcher; Fancyland LLC / Lattice OS\\[0.3em]
\small April 6, 2026\\[0.5em]
\small Status: Preprint\\[0.3em]
\small DOI: \href{https://doi.org/10.5281/zenodo.19442006}{10.5281/zenodo.19442006}\\[0.5em]

\small
MSC~2020: 11N05 (primary), 11A41, 82B05, 68T07 (secondary)\\[0.3em]
\textbf{Keywords:} prime numbers, Boltzmann distribution, softmax function, primorial moduli,\\
coprime residues, transition matrix, eigenvalue dynamics, spectral theory,\\
Chinese Remainder Theorem, information theory, arithmetic qubit\\[0.5em]

\textbf{Companion Papers:}\\[0.2em]
\small Active Transport in the Prime Gas \quad DOI: \href{https://doi.org/10.5281/zenodo.15127041}{10.5281/zenodo.15127041}\\
\small Testing Gauss's Class Number Conjecture with LLM-Guided BVP \quad DOI: \href{https://doi.org/10.5281/zenodo.15168862}{10.5281/zenodo.15168862}
\end{center}

"""

# Match from \begin{document} through end of companion papers block to blank line before abstract
title_pattern = re.compile(
    r'\\begin\{document\}\s*\n\s*\\section\{The Arithmetic Black Hole:.*?'
    r'\\textbf\{Companion papers:\}.*?\n\n',
    re.DOTALL
)
m = title_pattern.search(text)
if m:
    text = text[:m.start()] + new_title + text[m.end():]
    print("✓ Title block replaced")
else:
    print("✗ Title block NOT FOUND")

# ═══════════════════════════════════════════════════════════════════════
# 5. ABSTRACT ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════
text = re.sub(
    r'\\section\{Abstract\}\\label\{abstract\}\s*\n',
    r'\\begin{abstract}\n',
    text
)
# Insert \end{abstract} before "1. The Softmax-Boltzmann"
m = re.search(r'\n(\\section\{1\.)', text)
if m:
    text = text[:m.start()] + '\n\\end{abstract}\n' + text[m.start():]
    print("✓ Abstract wrapped in environment")

# ═══════════════════════════════════════════════════════════════════════
# 6. RE-INSERT TOC AFTER ABSTRACT
# ═══════════════════════════════════════════════════════════════════════
text = text.replace(
    '\\end{abstract}\n',
    '\\end{abstract}\n\n\\clearpage\n\\tableofcontents\n\\clearpage\n',
    1  # only first occurrence
)
print("✓ TOC inserted after abstract")

# ═══════════════════════════════════════════════════════════════════════
# 6. APPENDIX HANDLING
# ═══════════════════════════════════════════════════════════════════════
text = re.sub(r'(\\section\{Thermodynamic Stack)',
              r'\\appendix\n\1', text)
text = re.sub(r'\\section\{Appendix [A-D]: ', r'\\section{', text)
print("✓ \\appendix inserted, title prefixes stripped")

# ═══════════════════════════════════════════════════════════════════════
# 7. REMOVE PANDOC HORIZONTAL RULES
# ═══════════════════════════════════════════════════════════════════════
text = text.replace(
    r'\begin{center}\rule{0.5\linewidth}{0.5pt}\end{center}', ''
)
print("✓ Horizontal rules removed")

# ═══════════════════════════════════════════════════════════════════════
# 8. EQUATION NUMBERING: \[ → \begin{equation}, \] → \end{equation}
# ═══════════════════════════════════════════════════════════════════════
# Only match standalone display math (start of line or end of line)
text = re.sub(r'^(\s*)\\\[', r'\1\\begin{equation}', text, flags=re.MULTILINE)
text = re.sub(r'\\\](\s*)$', r'\\end{equation}\1', text, flags=re.MULTILINE)
eq_open = text.count(r'\begin{equation}')
eq_close = text.count(r'\end{equation}')
print(f"✓ Equations: {eq_open} open, {eq_close} close")

# ═══════════════════════════════════════════════════════════════════════
# 9. CENTER BRANDING
# ═══════════════════════════════════════════════════════════════════════
text = re.sub(
    r'(Prompt Studio \| Lattice OS \| Sovereign Computing.*?\n)',
    r'\\begin{center}\n\1\\end{center}\n',
    text
)
print("✓ Branding centered")

# ═══════════════════════════════════════════════════════════════════════
# 10. CLEARPAGE BEFORE APPENDIX C
# ═══════════════════════════════════════════════════════════════════════
text = re.sub(
    r'(\\section\{Summary of All\s*\nResults)',
    r'\\clearpage\n\1', text
)
print("✓ Clearpage before Appendix C")

# ═══════════════════════════════════════════════════════════════════════
# 11. TABLE FIXES
# ═══════════════════════════════════════════════════════════════════════

# 11a. §6 Demon's Ledger: Widen N column from 0.0658 to 0.1400
# The table has columns with specific widths; adjust N and compensate others
text = text.replace(
    r'* \real{0.0658}}',
    r'* \real{0.1400}}'
)
print("✓ §6 Demon's Ledger N column widened")

# 11b. §6.7 Four Laws: Widen Law column from 0.0769 to 0.1600
text = text.replace(
    r'* \real{0.0769}}',
    r'* \real{0.1600}}'
)
print("✓ §6.7 Four Laws Law column widened")

# 11c. Simplify minipage headers to inline in all longtables
# Replace \begin{minipage}[b]{\linewidth}\raggedright\nCONTENT\n\end{minipage}
# with just CONTENT
text = re.sub(
    r'\\begin\{minipage\}\[b\]\{\\linewidth\}\\raggedright\s*\n(.*?)\n\\end\{minipage\}',
    r'\1',
    text
)
print("✓ Minipage headers simplified to inline")

# 11d. Appendix C: Rebalance columns (Result: 0.38, Section: 0.12, Status: 0.50)
# Find the Appendix C table and adjust its column widths
# Original widths: 0.3200, 0.3600, 0.3200
appendix_c_pattern = re.compile(
    r'(\\section\{Summary of All\s*\nResults\}.*?)'
    r'\\real\{0\.3200\}\}(.*?)'
    r'\\real\{0\.3600\}\}(.*?)'
    r'\\real\{0\.3200\}\}',
    re.DOTALL
)
m = appendix_c_pattern.search(text)
if m:
    text = (text[:m.start()] + 
            m.group(1) + r'\real{0.3800}}' + m.group(2) + 
            r'\real{0.1200}}' + m.group(3) + r'\real{0.5000}}' +
            text[m.end():])
    print("✓ Appendix C columns rebalanced")
else:
    print("✗ Appendix C table not found for rebalancing")

# ═══════════════════════════════════════════════════════════════════════
# 12. APPENDIX B: Replace 3-column script longtable with 2-column
# ═══════════════════════════════════════════════════════════════════════

# Find the 3-column script longtable in Appendix B
# It starts with {\def\LTcaptype{none} and has "What it does" and "Runtime" columns
script_table_start = text.find(r'What it does and what it found')
if script_table_start > 0:
    # Find the {\def\LTcaptype{none} before it
    block_start = text.rfind(r'{\def\LTcaptype{none}', 0, script_table_start)
    # Find the closing } after \end{longtable}
    lt_end = text.find(r'\end{longtable}', script_table_start)
    block_end = text.find('}', lt_end + len(r'\end{longtable}'))
    
    if block_start > 0 and block_end > 0:
        old_table = text[block_start:block_end+1]
        
        new_table = r"""{\def\LTcaptype{none}
\begin{longtable}[]{@{}
  >{\raggedright\arraybackslash}p{(\linewidth - 2\tabcolsep) * \real{0.3200}}
  >{\raggedright\arraybackslash}p{(\linewidth - 2\tabcolsep) * \real{0.6800}}@{}}
\toprule\noalign{}
Script & What it does and what it found \\
\midrule\noalign{}
\endhead
\bottomrule\noalign{}
\endlastfoot
\texttt{compute\_boltzmann\_fit.py} & \textbf{The headline result.}
Sieves primes to \(N\), counts mod-\(m\) transitions, builds the forward
distance matrix (\(d(a,a) = m\)), computes the Boltzmann prediction at
\(T = N/\pi(N)\), reports \(R^2\), prints the full mod-30 matrix, and
measures Lemke Oliver--Soundararajan diagonal suppression. Includes
temperature convergence sweep and multi-modulus comparison. \\
\texttt{compute\_entanglement\_fixtures.py} & Peschel free-fermion
entanglement entropy at \(m = 6, 30, 210, 2310\). Produces exact
\(S_A(\gamma)\) fixtures for Module 9 validation. Discovers half-Page
saturation and inflection collapse. \\
\texttt{compute\_hawking\_page\_temperature.py} & Coarse
\(\gamma\)-sweep at \(m = 30030\) (19 points, \(\gamma \in [0, 1]\))
with checkpoint/resume. GPU MAGMA backend for \(30030\times 30030\) eigensolves.
Confirms 51.8\% Page saturation. \\
\texttt{compute\_fine\_grid\_m30030.py} & Fine \(\gamma\)-sweep at
\(m = 30030\) (25 points, \(\gamma \in [0.0001, 0.01]\)) targeting the
phase transition region. Confirms \(\gamma_c \to 0\) at production
scale. \\
\texttt{compute\_scrambler\_fixtures.py} & Module 7 scrambler fixture
generator --- eigenvalue spacings, KL divergences, Brody \(\omega\) for
the bulk \(H\)-spectrum at small moduli. \\
\texttt{compute\_scrambler\_sweep.py} & Full scrambler sweep measuring
\textbf{both} \(H\)-spectrum (bulk) and \(C_A\)-spectrum (boundary)
Brody \(\omega\) at \(m = 30, 210, 2310\) and \(m = 30030\). \\
\texttt{binary\_coupling\_experiment.py} & Binary coupling topology
proof. Replaces von Mangoldt \(\ln(p)\) weights with \(\{0,1\}\), sweeps
\(\gamma\) for \(m = 30, 210, 2310\). Proves holographic duality is
topological: 52/52 verdict match across \(m = 210\) and \(m = 2310\) (26
weighted + 26 binary). \\
\texttt{sweep\_small\_primorials.py} & Exact + fine entanglement sweeps
for \(m = 30, 210, 2310\). Produces high-resolution Page curves (201
points per modulus). \\
\texttt{verify\_freezeout.py} & Thermodynamic freeze-out verification.
Classifies moduli as thawed/frozen, measures trace convergence. \\
\texttt{algebraic\_vs\_transcendental.py} & Definitive test: algebraic
(\(-2\sqrt{3}/3\)) vs transcendental (\(-\ln(\pi)\)) at \(m = 30\).
Convergence direction, error bars. \\
\texttt{verify\_gemini\_nogo.py} & Tests competing scaling models:
power-law \((\ln N)^{-\beta}\) vs Hardy-Littlewood
\((\log\log N)^2/(\ln N)^2\). Verifies \(\beta = 2\gamma_w\) consistency
constraint (0.8\% gap). Confirms power-law fits dominate
(\(R^2 > 0.978\)). \\
\texttt{verify\_hyper\_radix.py} & Extracts full \(8\times 8\) eigenvalue spectrum
at \(m = 30\) across 29 \(N\)-values (\(10^4\)--\(10^9\)). Tracks
per-mode phase rotation rates: leading pair \(R^2 = 0.983\), sub-leading
pairs \(R^2 < 0.09\). Demonstrates coordinate invariance: phase rate
1.6\% error in bases \(e\), 3, 10, 2. Confirms only base 3 yields
rational \(1/\varphi(m_0) = 1/2\). \\
\texttt{hunt\_base\_mode.py} & Tests phase law at \(m = 210\) and
\(m = 2310\). Freeze-out diagnostic: \(m/T \geq 10\) at all accessible
\(N\) for \(m \geq 210\). CRT projection from
\(m = 210 \to m_0 = 6\) preserves trace to \(10^{-6}\). Identifies
leading eigenvalue at \(m = 210\) as sieve mode, not base CRT mode. \\
\texttt{eigenvector\_tracker.py} & Tracks eigenvalue modes by
eigenvector overlap (not magnitude rank) at \(m = 30\) and \(m = 210\).
At \(m = 30\): Mode \#0 rate \(+0.460\)/\(\log_3 T\), \(R^2 = 0.987\).
At \(m = 210\): Mode \#5 rate \(+0.443\), \(R^2 = 0.967\).
Resolves the mode-hopping problem. \\
\texttt{the\_well.py} & Reinterprets eigenvector tracker data in each
prime's native logarithmic base. Reveals the Hyper-Radix Tower:
\(p = 7\) fiber (1.3\% error), \(p = 5\) fiber (0.1\% error),
\(p = 3\) fiber (1.6\% error). \\
\texttt{complex\_waveform.py} & Complex eigenvalue extraction, phase
trajectory, damped oscillation fit, autocorrelation. Proves the
interference structure. \\
\end{longtable}
}

\textbf{Runtimes.} All CPU scripts run on a consumer PC (8 GB RAM, Python 3.10+, NumPy, SciPy) in 1--30 minutes each. The four \(m = 30030\) scripts (\texttt{compute\_hawking\_page\_temperature}, \texttt{compute\_fine\_grid\_m30030}, \texttt{compute\_scrambler\_sweep}, and the large-\(m\) path of \texttt{sweep\_small\_primorials}) require a GPU with \(\geq\) 24 GB VRAM and PyTorch with CUDA; tested on an NVIDIA L4 (GCE g2-standard-8), where each sweep takes 30--70 minutes.

\textbf{Code availability.} All scripts described in this section are
archived at
\url{https://github.com/fancyland-llc/arithmetic-black-hole}."""
        
        text = text.replace(old_table, new_table)
        print("✓ Appendix B script table: 3-col → 2-col (Runtime removed)")

# ═══════════════════════════════════════════════════════════════════════
# 13. APPENDIX B: Remove redundant B.1 Running the Scripts subsection
# ═══════════════════════════════════════════════════════════════════════
b1_pattern = re.compile(
    r'\\subsection\{B\.1 Running the Scripts\}.*?'
    r'archived at\s*\n\\url\{https://github\.com/fancyland-llc/arithmetic-black-hole\}\.',
    re.DOTALL
)
m = b1_pattern.search(text)
if m:
    text = text[:m.start()] + text[m.end():]
    print("✓ Appendix B.1 removed (redundant)")
else:
    print("  Appendix B.1 not found (may already be removed)")

# ═══════════════════════════════════════════════════════════════════════
# 14. APPENDIX B: Shorten verbatim script listing
# ═══════════════════════════════════════════════════════════════════════
old_verbatim_pattern = re.compile(
    r'\\begin\{verbatim\}\s*\n'
    r'compute_boltzmann_fit\.py\s+.*?'
    r'\\end\{verbatim\}',
    re.DOTALL
)
new_verbatim = r"""{\small
\begin{verbatim}
compute_boltzmann_fit.py            # Prime transition R^2, Boltzmann fit
compute_entanglement_fixtures.py    # Peschel entanglement entropy
compute_hawking_page_temperature.py # Coarse gamma-sweep at m=30030 (GPU)
compute_fine_grid_m30030.py         # Fine gamma-sweep, phase transition (GPU)
compute_scrambler_fixtures.py       # Scrambler spacings, KL, Brody omega
compute_scrambler_sweep.py          # Full H + C_A scrambler sweep (GPU)
verify_gemini_nogo.py               # KL-Frobenius consistency
verify_hyper_radix.py               # Eigenvalue spectrum, phase rates
hunt_base_mode.py                   # Multi-modulus phase hunt, CRT proj.
eigenvector_tracker.py              # Eigenvector continuity tracking
hyper_radix_tower.py                # Per-prime fiber modes
the_well.py                         # Tower in native prime bases
binary_coupling_experiment.py       # Binary coupling topology proof
sweep_small_primorials.py           # High-resolution Page curves
verify_freezeout.py                 # Thermodynamic freeze-out
algebraic_vs_transcendental.py      # Algebraic vs transcendental test
complex_waveform.py                 # Complex eigenvalue rotation
\end{verbatim}
}"""
m = old_verbatim_pattern.search(text)
if m:
    text = text[:m.start()] + new_verbatim + text[m.end():]
    print("✓ Verbatim block shortened with {\\small}")
else:
    print("✗ Verbatim block not found")

# ═══════════════════════════════════════════════════════════════════════
# FINAL CHECKS
# ═══════════════════════════════════════════════════════════════════════
fffd = text.count('\ufffd')
lines = text.count('\n')
print(f"\nFinal: {len(text)} chars, {lines} lines, FFFD={fffd}")

with open('ARITHMETIC_BLACK_HOLE.tex', 'w', encoding='utf-8', newline='\n') as f:
    f.write(text)
print("Written ARITHMETIC_BLACK_HOLE.tex")

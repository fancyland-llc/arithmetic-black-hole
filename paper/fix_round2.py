#!/usr/bin/env python3
"""Apply round-2 PDF fixes to ARITHMETIC_BLACK_HOLE.tex"""
import re

with open('ARITHMETIC_BLACK_HOLE.tex', 'r', encoding='utf-8') as f:
    text = f.read()
print(f"Read: {len(text)} chars")

# ═══════════════════════════════════════════════════════════════════════
# 1. PARAGRAPH SPACING: boost parskip
# ═══════════════════════════════════════════════════════════════════════
# The pandoc preamble uses parskip.sty which sets ~6pt.  Override after loading.
text = text.replace(
    r'\setcounter{secnumdepth}{-1}',
    '\\setcounter{secnumdepth}{-1}\n'
    '\\setlength{\\parskip}{6pt plus 2pt minus 1pt}'
)
print("✓ parskip set to 6pt")

# ═══════════════════════════════════════════════════════════════════════
# 2. EQUATION NUMBERING: section counter must be tracked manually
#    since secnumdepth=-1 means LaTeX doesn't auto-number sections,
#    so the section counter stays at 0, giving (0.1), (0.2) etc.
#    Fix: manually set the equation section counter at each \section.
# ═══════════════════════════════════════════════════════════════════════
# Extract section numbers from \section{N. Title} or \section{N Title}
# and insert \setcounter{section}{N} before each
def add_section_counters(text):
    lines = text.split('\n')
    new_lines = []
    for line in lines:
        # Match \section{N. or \section{N  where N is a number
        m = re.match(r'^\\section\{(\d+)[\.\s]', line)
        if m:
            sec_num = m.group(1)
            new_lines.append(f'\\setcounter{{section}}{{{sec_num}}}\\setcounter{{equation}}{{0}}')
        new_lines.append(line)
    return '\n'.join(new_lines)

text = add_section_counters(text)
eq_lines = [i for i, l in enumerate(text.split('\n')) if 'setcounter{section}' in l and 'setcounter{equation}' in l]
print(f"✓ Equation numbering: inserted {len(eq_lines)} section counter resets")

# ═══════════════════════════════════════════════════════════════════════
# 3. TABLE 6.7 (Four Eigenvalue Laws): shrink to fit margins
#    Sum of reals: 0.16+0.2462+0.2923+0.20+0.1077+0.16 = 1.1662 > 1.0!
#    That's why it overflows. Fix the column widths to sum to ~1.0
# ═══════════════════════════════════════════════════════════════════════
text = text.replace(
    r""">{\raggedright\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.1600}}
  >{\raggedright\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.2462}}
  >{\raggedright\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.2923}}
  >{\raggedright\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.2000}}
  >{\raggedright\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.1077}}
  >{\raggedright\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.1600}}@{}}""",
    r""">{\raggedright\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.1200}}
  >{\raggedright\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.2000}}
  >{\raggedright\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.2600}}
  >{\raggedright\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.1200}}
  >{\raggedright\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.0800}}
  >{\raggedright\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.0700}}@{}}"""
)
print("✓ §6.7 Four Eigenvalue Laws table: columns fit within margins")

# ═══════════════════════════════════════════════════════════════════════
# 4. BINARY COUPLING TABLE: shrink to fit
#    Sum: 0.14+0.1711+0.1974+0.25+0.2237+0.0921 = 1.0743 > 1.0
# ═══════════════════════════════════════════════════════════════════════
text = text.replace(
    r""">{\raggedright\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.1400}}
  >{\raggedright\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.1711}}
  >{\raggedright\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.1974}}
  >{\raggedright\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.2500}}
  >{\raggedright\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.2237}}
  >{\raggedright\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.0921}}@{}}""",
    r""">{\raggedright\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.0600}}
  >{\raggedright\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.0800}}
  >{\raggedright\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.1000}}
  >{\raggedright\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.2200}}
  >{\raggedright\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.2200}}
  >{\raggedright\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.0800}}@{}}"""
)
print("✓ Binary Coupling table: columns fit within margins")

# ═══════════════════════════════════════════════════════════════════════
# 5. SCRIPT TABLE: widen Script column, use smaller font
#    Current: 0.32 / 0.68.  Change to tabularx or just use \small
#    and adjust widths to give more room to col 1
# ═══════════════════════════════════════════════════════════════════════
text = text.replace(
    r"""{\def\LTcaptype{none}
\begin{longtable}[]{@{}
  >{\raggedright\arraybackslash}p{(\linewidth - 2\tabcolsep) * \real{0.3200}}
  >{\raggedright\arraybackslash}p{(\linewidth - 2\tabcolsep) * \real{0.6800}}@{}}
\toprule\noalign{}
Script & What it does and what it found \\""",
    r"""{\def\LTcaptype{none}
\small
\begin{longtable}[]{@{}
  >{\ttfamily\raggedright\arraybackslash}p{(\linewidth - 2\tabcolsep) * \real{0.3600}}
  >{\raggedright\arraybackslash}p{(\linewidth - 2\tabcolsep) * \real{0.6400}}@{}}
\toprule\noalign{}
\textnormal{Script} & \textnormal{What it does and what it found} \\"""
)
# Also strip \texttt{} wrappers from script names since column is now \ttfamily
text_after_table = text.find(r'\textnormal{Script}')
table_end = text.find(r'\end{longtable}', text_after_table)
table_section = text[text_after_table:table_end]
# Replace \texttt{name} with just name (since column is ttfamily)
table_section_fixed = re.sub(r'\\texttt\{([^}]+)\}', r'\1', table_section)
text = text[:text_after_table] + table_section_fixed + text[table_end:]
print("✓ Script table: wider col 1 (0.36), \\small font, \\ttfamily column")

# ═══════════════════════════════════════════════════════════════════════
# 6. BRANDING: centered with cosmetic line above
# ═══════════════════════════════════════════════════════════════════════
text = text.replace(
    "\\emph{Fancyland LLC --- Lattice OS research infrastructure.}\n\n\\emph{The rabbit has been caught.}",
    "\\vfill\n\\begin{center}\n\\rule{0.4\\linewidth}{0.4pt}\\\\[0.8em]\n"
    "\\emph{Fancyland LLC --- Lattice OS research infrastructure.}\\\\[0.3em]\n"
    "\\emph{The rabbit has been caught.}\n\\end{center}"
)
print("✓ Branding: centered with cosmetic rule")

# ═══════════════════════════════════════════════════════════════════════
# WRITE
# ═══════════════════════════════════════════════════════════════════════
fffd = text.count('\ufffd')
lines = text.count('\n')
print(f"\nFinal: {len(text)} chars, {lines} lines, FFFD={fffd}")

with open('ARITHMETIC_BLACK_HOLE.tex', 'w', encoding='utf-8', newline='\n') as f:
    f.write(text)
print("Written ARITHMETIC_BLACK_HOLE.tex")

# Rules for agents working in this repo

- Do not guess what a script does. Before running any script you are not certain about, read it and work out its effect, including what it writes, overwrites, kills or sends to the cluster. Only then run it.
- Never cancel, kill or resubmit the user's jobs (scancel, kill, pkill, sbatch) unless the user explicitly asks for that exact action in that message. "I need to kick these out" is a plan, not an instruction.
- Never run greatlakes/sync_to_greatlakes.sh while any of the user's jobs are running. It pushes the git-tracked greatlakes/slurm-*.out files over the live job logs and kills the jobs. Check `squeue -u <user>` first; push single files with scp.
- Do not delete or overwrite anything on the cluster or in results directories without the user's approval.

## Paper wording (paper/Mortis)

- Make the smallest fix. If one word fixes the sentence (add "The"), change only that word. Do not restructure, and do not invent problems the sentence does not have. A sentence that states a purpose is never replaced by one that states a process; add the new sentence instead.
- Cut every dispensable word: no fillers ("Concretely"), no restating the previous sentence, no flourishes. Short plain sentences.
- Never translate the author's casual Chinese remarks word for word into the paper; write the idea properly.
- Use plain words. No jargon the paper never defined ("support", "pooled counts", "back-off"); use the symbol already defined instead ($n_k$, not "support"). No compressed noun phrases ("a call due now" → "an imminent call"), no lyrical phrasing ("realizing this vision"), no role-play verbs ("discovering", "identifying").
- One term per concept within a paragraph; do not drift between level/context/row or group/category. Never merge two of the author's terms into a phrase that means nothing ("information has structure"). Standard terms are fine when exact ("zero in-degree").
- One symbol per concept and one concept per symbol; never reuse a letter for two things. Keep notation identical between the text and the algorithms ($L_q$, $q\in Q$). Replace an "it" whose referent is unclear with the noun.
- Define every term and symbol where it is first used; "these requirements" needs an antecedent. Define with a short "where:" list, not stacked relative clauses.
- Pseudocode: math notation, no English prose lines ("Put $b$ into …"). Define every symbol, including the nodes and edges of derived graphs and each key's domain. Write a loop out instead of calling an undefined helper and explaining it in a comment. Comments are short and exact. Use common notation ([ ] list, (a, b) tuple) and do not assume readers share a convention.
- State things positively. No double negatives ("cached on neither tier" → "that must be prefilled"), no strawman absolutes ("carries no intent", "solely", "without any"), no defensive clauses that pre-empt every objection.
- Keep the author's emphasis and meaning: keep the subject they chose, keep restrictive clauses restrictive, keep linking openers such as "To …". Avoid tongue-twister repeats ("by … by"). A bridge sentence picks up the noun the previous sentence ended on.
- When describing other systems' limits, use neutral words ("information") that do not leak our own design.
- Reply with the revised sentence and one or two sentences of reason. No lists or option menus.
- Before explaining a mechanism, check the code in static_analysis/, server/ and model/, not sglang/. Text and pseudocode must match that code exactly.
- Do not edit the LaTeX source unless asked for that change. Compiling is always fine: activate the `mortis` conda env and run `latexmk -pdf mortis.tex` in paper/Mortis (the system pdflatex lacks algorithm.sty). The main text must end by page 10.

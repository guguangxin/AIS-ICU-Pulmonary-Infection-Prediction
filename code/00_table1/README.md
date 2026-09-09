# Main Table 1 and Supplementary Table S32

`Table1_baseline_characteristics.py` reproduces the final presentation split requested during peer review.

- **Main Table 1** retains demographic/comorbidity information, the primary first-48-hour support/treatment predictors (surgery, mechanical ventilation, intubation/tracheotomy, diuretics), and laboratory variables.
- **Supplementary Table S32** contains the additional first-48-hour intervention/medication candidate variables moved out of the main table for readability: CVC, CRRT, immunosuppressants, anticholinergics, cholinesterase inhibitors, benzodiazepines, non-benzodiazepine sedatives, antipsychotics, vasoactive agents, and BSA.

Run:

```bash
python code/00_table1/Table1_baseline_characteristics.py \
  --input /path/to/authorized/analysis_ready.csv \
  --output-dir outputs/table1
```

The participant-level input and generated CSV outputs are local/restricted and are excluded by the repository `.gitignore`.

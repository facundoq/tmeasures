4. Redesign TransformationSet, make it just a list of transformations, remove the parameters and valid_transformation methods. Move parameters to tmexp.
5. Add documentation to classes and functions
6. Generate html docs from docstrings
7. Fix experiments on other repo to use new API
8. Fix non-deep names in AutoActivations 
9.  Label font size for visualization must depend on layer name. Else, use last part of layer name or use type.
10. Fix errors propagating correctly when a thread crashes
11. Add labels to TQDM bars to indicate the measure being computed (use abbreviation)
12. Re-add Distance implementations
13. Add Numpy-Pytorch adapter. 
14. Check Numpy and Pytorch implementations match in their results
15. Add graph or tree visualization (or at least a hierarchical visualization) using colors to indicate invariance

### How can you contribute?

New and exciting objectives appear in NLP research often, and the Adaptor library aims to make it as simple as possible to add them! If you'd like to add a new `Objective` in `Adaptor` follow these steps:

1. **Implement it**: pick the logically-best-matching abstract objective from `objectives`,
and implement the remaining abstract methods.
2. **Test it**: add a simple test for your objective to `tests/objectives_test.py`, 
that will pass `assert_module_objective_ok`.
3. **End-to-end-test it**: add a test to `end2end_usecases_test.py` with a complete 
demonstration on how to use your objective in a meaningful way
4. (optional): **Create an example** that will apply the objective in a real training process, on real data. 
See other examples in `examples` folder.
5. **Share!** Create a PR or issue here in GitHub with a link to your fork, and we'll happily take a look!

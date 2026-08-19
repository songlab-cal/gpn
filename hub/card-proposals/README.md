# Review-only Hugging Face documentation proposals

These files are proposed replacements or templates, not mirrors of live Hub cards.
Nothing in this directory is published automatically.

The five canonical model proposals use the immutable revisions validated in
`tests/fixtures/published_model_baseline.json`. The shared GPN-Star template covers
the other public family checkpoints without claiming that each has an approved
numerical regression.

Before any external write:

1. resolve every `TODO(maintainer)`;
2. review scientific inputs, outputs, score direction, and citations;
3. confirm the exact target repository and current head;
4. explicitly authorize the model-card or collection write; and
5. rerun both the metadata audit and, for a supported-model revision change, the
   opt-in published-model tests.

Do not copy model implementations into cards. The installed `gpn` package remains
canonical; cards should use explicit registration and standard Transformers
AutoClasses.

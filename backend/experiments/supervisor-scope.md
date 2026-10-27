# Supervisor scope prompt experiment

This experiment tests whether neutral, question-grounded delegation and evidence-grounded synthesis reduce unnecessary expansion and correction work while retaining complete answers. The only candidate business change is in rules 4 and 7 of the Supervisor system prompt. Researcher, Reviewer, tool schemas, retrieval, Memory and provider settings remain the same.

The shared experimental entry supports two strict contracts. The existing exact-body deduplication experiment keeps one source tree with deduplication off in A and on in B. The Supervisor scope experiment uses separately frozen A/B source trees with deduplication on in both. Its source and prompt identities must match the manifest, and the business differences must be exactly the approved Supervisor prompt replacement. An unknown experiment kind, a different switch combination or a second business change is rejected.

Each candidate needs an isolated checkout, runtime, database, outputs, temporary files and model/source identities. Validation of imports must exercise the actual isolated Python child entry: changing sys.path after a package has been loaded does not select a different variant. A model task cannot write its own pair-review approval; controller review records live outside its writable runtime. Reviews bind the actual answer, sources and receipts, and their elapsed time remains part of the batch deadline.

The experimental controls preserve pre-send reservations, cumulative prior usage, complete actual input/output accounting, shared root-family budgets, absolute process-group deadlines and no blind retry of unknown receipts. Offline tests use synthetic responses and separate temporary stores. They do not establish remote model acceptance, scientific correctness or an efficiency improvement.

Private questions, paper files, traces, grading records, deployment configuration and credentials are not part of this repository document. Real execution requires separately reviewed sample, resource, scoring and budget manifests plus explicit authorization. No experiment authorizes a deployment switch or merge by itself.

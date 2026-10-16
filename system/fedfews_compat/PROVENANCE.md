# FedFewS standard compatibility sources

Upstream: [pgg3/FedFew](https://github.com/pgg3/FedFew), commit
`1b29dec70b66d89b956d90e0da024897f698897a`, MIT, copyright Ping Guo.
The complete upstream license is retained in `vendor/LICENSE`.

On 2026-10-16, a fresh, clean clone was obtained at
`D:\All_Git_Repo\FedFew_reference`. There was no actual local FedFew working
copy; the user confirmed this and authorized using GitHub instead. The earlier
audit copies under the user's Documents directory are not the reference for
this migration. No claim is made about unrecorded edits on the training server.

`source_manifest.py` records raw SHA256 values of all relevant upstream files,
including the parser, YAMLs and runner. Raw hashes can differ across checkouts
with CRLF/LF conversion; acceptance compares executed behavior and parser/config
values, and records the actual checkout hashes/status rather than hiding this.

## Adaptation boundaries

* `vendor/clientbase.py`, `clientfedfews.py`, `serverbase.py`,
  `serverfedfews.py`: original bodies, with package imports changed to relative
  imports; trailing whitespace is normalized. The disabled DLG import resolves to an explicit rejecting stub.
  No training, selection, weighting, aggregation, or RNG statements were changed.
* `vendor/data_utils.py`: original reader/decoder. `PFLLIB_DATA_DIR` supplies an
  independent root. `.pkl` takes precedence over `.npz`, including when both
  exist. No image transformations or normalization were added.
* `vendor/models.py`: the original `FedAvgCNN` class (trailing whitespace normalized); unrelated models
  and their imports omitted. Parameter names/order, constructor order and
  initialization are preserved. CIFAR-100 has 924,708 parameters.
* `vendor/arguments.py`: original `str2bool` and parser declarations extracted
  with trailing whitespace normalized. `config.py` implements only the supported runner mapping;
  acceptance also executes the actual external runner's merge and command builder.
  Upstream's unused `momentum=.1` default remains visible, but its standard SGD
  constructors really use momentum=0 and weight_decay=0. No optimizer override
  was introduced.
* `runtime.py`: calls the copied original `FedFewS.train()` inclusive loop. Hooks
  observe actual calls/sampler indices, export diagnostics, and redirect outputs.
  Upstream evaluation consumes RNG; that consumption is retained. A sampler is
  never iterated twice to log its batches. New diagnostics perform no random draws.
* Output adaptation: upstream's inherited `save_global_model()` saves the unused
  single template, not `global_model_set`. Our hook saves the actual K aggregated
  servers, every client's K post-local models, deployed index, optimizers,
  schedulers and final RNG. Existing H5 fields are retained and extended. This is
  an output correction only; it does not alter any training or aggregation step.
* UTF-8 console/log handling permits upstream emoji output on Windows GBK consoles.

## External source reference

`reference.py` executes files from the separately supplied `--source-root`.
It only resolves package imports into isolated namespaces, extracts the CNN and
parser, and substitutes a rejecting DLG stub. It does not use migrated classes
for client/server/model/reader calculations. The original external
`FedFewS.train()` runs, including its evaluation and non-evaluation branches.
Both sides receive the same initial Python/NumPy/Torch CPU/CUDA states and the
same observation/output hooks. A separate test compares the observed path to an
uninstrumented original-body path, including all final local/server parameters
and RNG states.

No FedClip Client/Server base, FedTargetProj entry, model or default was changed.
No environment was copied or modified. This is not a heterogeneous low-rank
FedFew implementation.

# Third-Party Software Notices and Attributions

The original code in this repository is released by the authors under the MIT
License (see [LICENCE](LICENCE)). The experiments additionally build on two
third-party projects. **Neither project's source code is redistributed in this
repository.** Each is downloaded directly from its official upstream repository
at a pinned commit by a `setup_*.sh` script, and our changes are shipped only as
**patch (diff) files** that are applied on top of the freshly cloned source.
Each upstream project remains governed by its own license.

This file provides the attributions required by those upstream licenses.

---

## 1. Savanna (Evo 2 experiments)

- **Used by:** `pretraining_evo2/`.
- **Upstream:** https://github.com/Zymrael/savanna
- **Retrieved at setup by:** `pretraining_evo2/setup_savanna.sh`, which clones
  the upstream repository at the commit recorded in
  `pretraining_evo2/BASE_COMMIT.txt` and applies the patches in
  `pretraining_evo2/patches/`.
- **License:** Apache License, Version 2.0
  (http://www.apache.org/licenses/LICENSE-2.0).

Per the upstream `NOTICE` file, the following copyright attributions are
retained:

```
Copyright 2024 Arc Institute. All rights reserved
Copyright 2024 Michael Poli. All rights reserved
Copyright 2024 Stanford University. All rights reserved

This software is licensed under the Apache License, Version 2.0.
```

In accordance with Section 4(b) of the Apache License 2.0, we note that **files
modified by us are distributed as patches** (`pretraining_evo2/patches/*.patch`)
rather than as modified copies of the original source; applying a patch produces
a clearly marked, traceable change relative to the pinned upstream commit. The
full text of the Apache License 2.0 is available at the URL above and in the
upstream repository's `LICENSE` file.

---

## 2. GENERator (GENERator-800M experiments)

- **Used by:** `pretraining_GENERator/`.
- **Upstream:** https://github.com/GenerTeam/GENERator
- **Retrieved at setup by:** `pretraining_GENERator/setup_generator.sh`, which
  clones the upstream repository at commit
  `44b0bda48676b6362ba9f58b648c6893f34907a6` and applies
  `pretraining_GENERator/custom_trainer.py.patch`.
- **License:** MIT License.

The MIT License requires that the following copyright and permission notice be
retained. It applies to the upstream GENERator software (obtained at setup time)
and to the portions of it reproduced in our patch file:

```
MIT License

Copyright (c) 2025 GenerTeam

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

Because the upstream sources are fetched (not vendored) and only diffs are
distributed here, this repository does not place your downstream use under the
Apache License; the cloned Savanna tree you create at setup time remains subject
to the Apache License 2.0, and the cloned GENERator tree remains subject to the
MIT License above. Both upstream licenses are permissive and compatible with the
MIT License under which our own code is released.
